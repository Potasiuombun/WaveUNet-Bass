#!/usr/bin/env python3
"""Stage-3 training script for VAE WaveUNet."""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.serialized import collate_frame_batch, get_dataset_info, load_serialized_dataset
from src.losses.reconstruction import L1Loss, NMSELoss
from src.losses.spectral import MultiResolutionSTFTLoss
from src.models.factory import create_model
from src.training.callbacks import EarlyStopper
from src.training.checkpointing import Checkpoint
from src.training.metrics import mae, nmse
from src.utils.logging import CSVLogger, JSONLogger
from src.utils.seed import set_seed


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config from disk."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize config keys for stage-3 VAE training."""
    cfg = dict(config)
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("training", {})

    data_cfg = cfg["data"]
    data_cfg.setdefault("legacy_val_ratio", 0.15)

    if "name" in cfg["model"] and "type" not in cfg["model"]:
        cfg["model"]["type"] = cfg["model"]["name"]
    cfg["model"].setdefault("type", "vae_waveunet")
    cfg["model"].setdefault("activation", "leaky_relu")
    cfg["model"].setdefault("output_activation", None)
    cfg["model"].setdefault("latent_channels", 64)

    loss_cfg = cfg["loss"]
    loss_cfg.setdefault("alpha_reconstruction", 1.0)
    loss_cfg.setdefault("gamma_mrstft", 1.0)
    loss_cfg.setdefault("beta_kl", 1e-4)
    loss_cfg.setdefault("beta_kl_start", float(loss_cfg["beta_kl"]))
    loss_cfg.setdefault("beta_kl_end", float(loss_cfg["beta_kl"]))
    loss_cfg.setdefault("kl_warmup_epochs", 0)
    loss_cfg.setdefault("l1_weight", 1.0)
    loss_cfg.setdefault("nmse_weight", 1.0)
    loss_cfg.setdefault("mrstft_fft_sizes", [1024, 512, 256])
    loss_cfg.setdefault("mrstft_hop_sizes", [512, 256, 128])

    # KL schedule — structured section that overrides legacy loss keys when present.
    if "kl_schedule" not in cfg:
        cfg["kl_schedule"] = {"type": "linear_warmup"}
    kl_sched = cfg["kl_schedule"]
    kl_sched.setdefault("type", "linear_warmup")
    kl_sched.setdefault("beta_start", 0.0)
    kl_sched.setdefault("beta_max", 1e-4)
    kl_sched.setdefault("warmup_epochs", 0)
    kl_sched.setdefault("num_cycles", 4)
    kl_sched.setdefault("cycle_ratio", 0.5)

    # Free bits — optional per-dimension KL floor to prevent posterior collapse.
    if "kl_free_bits" not in cfg:
        cfg["kl_free_bits"] = {"enabled": False, "value": 0.0, "mode": "per_dim"}
    kl_fb = cfg["kl_free_bits"]
    kl_fb.setdefault("enabled", False)
    kl_fb.setdefault("value", 0.0)
    # mode: "per_dim" applies the free-bits floor per latent dimension (recommended);
    # "off" disables free bits regardless of the enabled flag.
    kl_fb.setdefault("mode", "per_dim")

    train_cfg = cfg["training"]
    train_cfg.setdefault("batch_size", 16)
    train_cfg.setdefault("num_workers", 2)
    # Legacy serialized .pkl datasets can consume high RAM with worker processes.
    # Keep a safer default of 0 workers for legacy row-level splitting.
    train_cfg.setdefault("legacy_num_workers", 0)
    train_cfg.setdefault("num_epochs", 10)
    train_cfg.setdefault("optimizer", "adamw")
    train_cfg.setdefault("lr", 5e-5)
    train_cfg.setdefault("weight_decay", 1e-4)
    train_cfg.setdefault("gradient_clip", 1.0)
    train_cfg.setdefault("early_stopping_patience", 5)
    train_cfg.setdefault("checkpoint_dir", "checkpoints/vae_stage3")
    train_cfg.setdefault("warm_start_checkpoint", None)

    if "logging" not in cfg:
        exp = cfg.get("experiment_name", "vae_stage3")
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
        print("CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    if device not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported device '{requested_device}'. Use 'cuda' or 'cpu'.")
    return device


def resolve_existing_path(path_str: Optional[str], config_path: str) -> Optional[str]:
    """Resolve path from absolute, cwd-relative, or config-relative input."""
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


def build_train_val_loaders(dataset_file: str, cfg: Dict[str, Any], seed: int):
    """Build train/val loaders with a memory-safe path for legacy datasets.

    For datasets that already contain a ``split`` column, preserves the existing
    behavior by loading split-specific DataLoaders directly.

    For legacy datasets (no split metadata), loads a single base dataset once
    and applies deterministic row-level train/val ``Subset`` splits. This avoids
    holding two full copies of the DataFrame in memory.
    """
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["training"]["num_workers"])
    info = get_dataset_info(dataset_file)

    if bool(info.get("has_split_column", False)):
        train_loader = load_serialized_dataset(
            dataset_file,
            split="train",
            batch_size=batch_size,
            num_workers=num_workers,
            validate=True,
        )
        val_loader = load_serialized_dataset(
            dataset_file,
            split="val",
            batch_size=batch_size,
            num_workers=num_workers,
            validate=False,
        )
        return train_loader, val_loader

    print("  Legacy dataset without split column detected. Using deterministic row-level split.")
    base_dataset = load_serialized_dataset(
        dataset_file,
        split=None,
        batch_size=None,
        num_workers=0,
        validate=True,
    )

    total = len(base_dataset)
    if total < 2:
        raise ValueError("Need at least 2 rows in legacy dataset for train/val split.")

    val_ratio = float(cfg.get("data", {}).get("legacy_val_ratio", 0.15))
    val_ratio = max(0.01, min(0.49, val_ratio))
    n_val = max(1, int(total * val_ratio))
    n_train = total - n_val
    if n_train < 1:
        raise ValueError("Legacy split produced empty train set. Lower data.legacy_val_ratio.")

    gen = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(total, generator=gen).tolist()
    val_indices = perm[:n_val]
    train_indices = perm[n_val:]

    legacy_workers = int(cfg["training"].get("legacy_num_workers", 0))
    if legacy_workers != num_workers:
        print(f"  Using legacy_num_workers={legacy_workers} (configured num_workers={num_workers}).")
    print(f"  Legacy row split: train={n_train} val={n_val} (val_ratio={val_ratio:.3f})")

    train_loader = DataLoader(
        Subset(base_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=legacy_workers,
        collate_fn=collate_frame_batch,
    )
    val_loader = DataLoader(
        Subset(base_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=legacy_workers,
        collate_fn=collate_frame_batch,
    )
    return train_loader, val_loader


def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer from training config."""
    train_cfg = cfg["training"]
    opt_name = str(train_cfg["optimizer"]).lower()
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])

    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{opt_name}'")


def kl_divergence_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits: float = 0.0,
) -> torch.Tensor:
    """Compute KL divergence with optional free bits per latent dimension.

    Standard KL (``free_bits <= 0``):
        KL = mean(-0.5 * (1 + logvar - mu^2 - exp(logvar)))

    Free bits (``free_bits > 0``):
        Per-dimension KL is averaged over batch and time, then floored at
        *free_bits* before summing over latent dimensions.  This prevents the
        model from being penalized for dimensions it actively uses (those below
        the floor) while still regularizing over-used ones.

        KL_dim[c] = mean_{b,t}(-0.5*(1 + logvar_{b,c,t} - mu_{b,c,t}^2 - exp(logvar_{b,c,t})))
        KL = sum_c max(free_bits, KL_dim[c])

    Args:
        mu: Posterior mean, shape [B, C, T].
        logvar: Posterior log-variance (clamped), shape [B, C, T].
        free_bits: Minimum KL per latent dimension.  0 disables (standard KL).

    Returns:
        Scalar KL loss.
    """
    kl_per_element = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())  # [B, C, T]

    if free_bits <= 0.0:
        return kl_per_element.mean()

    # Mean over batch + time per latent channel → [C]; floor; sum over C.
    kl_per_dim = kl_per_element.mean(dim=[0, 2])  # [C]
    return torch.clamp(kl_per_dim, min=float(free_bits)).sum()


def _beta_kl_linear(epoch: int, beta_start: float, beta_end: float, warmup_epochs: int) -> float:
    """One-shot linear warmup: rises from beta_start to beta_end over warmup_epochs."""
    if warmup_epochs <= 0:
        return float(beta_end)
    progress = min(max(epoch - 1, 0), warmup_epochs) / float(warmup_epochs)
    return float(beta_start + (beta_end - beta_start) * progress)


def _beta_kl_cyclical(
    epoch: int,
    num_epochs: int,
    num_cycles: int,
    beta_start: float,
    beta_max: float,
    cycle_ratio: float,
) -> float:
    """Cyclical annealing schedule (Fu et al., 2019).

    Divides the training run into *num_cycles* equal cycles.  Within each
    cycle the schedule has two phases:

    1. **Ramp** (first ``cycle_ratio`` of the cycle): beta rises linearly
       from *beta_start* to *beta_max*.
    2. **Plateau** (remaining fraction): beta stays at *beta_max*.

    At the start of each new cycle beta resets to *beta_start*, giving the
    decoder repeated opportunities to rebuild reconstruction quality before
    KL pressure increases again.
    """
    cycle_ratio = float(max(0.0, min(1.0, cycle_ratio)))
    cycle_len = max(1.0, num_epochs / max(1, num_cycles))
    t = epoch - 1  # convert to 0-indexed
    cycle_pos = (t % cycle_len) / cycle_len  # position within current cycle in [0, 1)

    if cycle_ratio <= 0.0:
        return float(beta_max)
    if cycle_pos < cycle_ratio:
        progress = cycle_pos / cycle_ratio
        return float(beta_start + (beta_max - beta_start) * progress)
    return float(beta_max)


def compute_beta_kl(cfg: Dict[str, Any], epoch: int, num_epochs: int) -> float:
    """Dispatch KL weight computation based on ``cfg['kl_schedule']``.

    Reads ``cfg["kl_schedule"]["type"]`` and delegates to the appropriate
    helper.  Falls back to ``linear_warmup`` using legacy ``cfg["loss"]`` keys
    when the structured section is absent.

    Supported schedule types:
    - ``constant``: fixed ``beta_max`` for all epochs.
    - ``linear_warmup`` (default): one-shot linear ramp from ``beta_start`` to
      ``beta_max`` over ``warmup_epochs``, then stays at ``beta_max``.
    - ``cyclical``: cyclical annealing (Fu et al., 2019).
    """
    sched = cfg.get("kl_schedule", {})
    sched_type = str(sched.get("type", "linear_warmup")).lower()

    if sched_type == "constant":
        return float(sched.get("beta_max", cfg.get("loss", {}).get("beta_kl", 1e-4)))

    if sched_type == "cyclical":
        return _beta_kl_cyclical(
            epoch=epoch,
            num_epochs=num_epochs,
            num_cycles=int(sched.get("num_cycles", 4)),
            beta_start=float(sched.get("beta_start", 0.0)),
            beta_max=float(sched.get("beta_max", 1e-4)),
            cycle_ratio=float(sched.get("cycle_ratio", 0.5)),
        )

    if sched_type == "linear":
        # Structured linear schedule using kl_schedule fields (preferred over legacy loss keys).
        return _beta_kl_linear(
            epoch=epoch,
            beta_start=float(sched.get("beta_start", 0.0)),
            beta_end=float(sched.get("beta_max", 1e-4)),
            warmup_epochs=int(sched.get("warmup_epochs", 0)),
        )

    # Default: one-shot linear warmup via legacy loss-section keys ("linear_warmup").
    loss_cfg = cfg.get("loss", {})
    return _beta_kl_linear(
        epoch=epoch,
        beta_start=float(loss_cfg.get("beta_kl_start", loss_cfg.get("beta_kl", 1e-4))),
        beta_end=float(loss_cfg.get("beta_kl_end", loss_cfg.get("beta_kl", 1e-4))),
        warmup_epochs=int(loss_cfg.get("kl_warmup_epochs", 0)),
    )


def warm_start_from_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str) -> Optional[int]:
    """Warm-start model weights with non-strict loading.

    Returns loaded epoch when available.
    """
    state = torch.load(checkpoint_path, map_location=device)
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state
    load_result = model.load_state_dict(model_state, strict=False)

    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    if missing:
        print(f"Warm-start missing keys: {len(missing)}")
    if unexpected:
        print(f"Warm-start unexpected keys: {len(unexpected)}")

    return int(state.get("epoch", 0)) if isinstance(state, dict) else None


def train_one_epoch(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: str,
    l1_loss: L1Loss,
    nmse_loss: NMSELoss,
    mrstft_loss: MultiResolutionSTFTLoss,
    cfg: Dict[str, Any],
    beta_kl: float,
    free_bits: float = 0.0,
) -> Dict[str, float]:
    """Run one training epoch and return averaged metrics.

    Logged keys (prefixed with ``train_``):
    - ``reconstruction``: weighted recon loss (L1 + NMSE).
    - ``mrstft``: multi-resolution STFT loss.
    - ``kl_raw``: KL *before* free-bits thresholding (standard mean over all elements).
    - ``kl``: KL *after* free-bits (used in backprop; equals ``kl_raw`` when disabled).
    - ``kl_weighted``: beta * kl (actual KL contribution to total loss).
    - ``l1``, ``nmse``: individual reconstruction components.
    - ``total``: final combined loss.
    """
    model.train()
    totals = {
        "total": 0.0,
        "reconstruction": 0.0,
        "mrstft": 0.0,
        "kl_raw": 0.0,
        "kl": 0.0,
        "kl_weighted": 0.0,
        "l1": 0.0,
        "nmse": 0.0,
    }
    num_batches = 0

    alpha = float(cfg["loss"]["alpha_reconstruction"])
    gamma = float(cfg["loss"]["gamma_mrstft"])
    l1_w = float(cfg["loss"]["l1_weight"])
    nmse_w = float(cfg["loss"]["nmse_weight"])
    grad_clip = float(cfg["training"]["gradient_clip"])

    for batch in train_loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        recon = outputs["recon"]
        mu = outputs["mu"]
        logvar = outputs["logvar"]

        l1_term = l1_loss(recon, targets)
        nmse_term = nmse_loss(recon, targets)
        recon_term = l1_w * l1_term + nmse_w * nmse_term
        mrstft_term = mrstft_loss(recon, targets)
        # kl_raw: standard mean KL (no free bits), used for monitoring.
        kl_raw = kl_divergence_loss(mu, logvar, free_bits=0.0)
        # kl_term: KL used in backprop, applies free bits when enabled.
        kl_term = kl_divergence_loss(mu, logvar, free_bits=free_bits)

        total = alpha * recon_term + gamma * mrstft_term + beta_kl * kl_term
        total.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        totals["total"] += float(total.item())
        totals["reconstruction"] += float(recon_term.item())
        totals["mrstft"] += float(mrstft_term.item())
        totals["kl_raw"] += float(kl_raw.item())
        totals["kl"] += float(kl_term.item())
        totals["kl_weighted"] += beta_kl * float(kl_term.item())
        totals["l1"] += float(l1_term.item())
        totals["nmse"] += float(nmse_term.item())
        num_batches += 1

    return {f"train_{k}": v / max(1, num_batches) for k, v in totals.items()}


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    val_loader,
    device: str,
    l1_loss: L1Loss,
    nmse_loss: NMSELoss,
    mrstft_loss: MultiResolutionSTFTLoss,
    cfg: Dict[str, Any],
    beta_kl: float,
    free_bits: float = 0.0,
) -> Dict[str, float]:
    """Run one validation epoch and return averaged metrics.

    Logged keys (prefixed with ``val_``):
    - ``reconstruction``: weighted recon loss.
    - ``mrstft``: STFT loss.
    - ``kl_raw``: KL before free-bits thresholding.
    - ``kl``: KL after free-bits (matches backprop KL).
    - ``kl_weighted``: beta * kl.
    - ``l1``, ``nmse``: reconstruction components.
    - ``nmse_metric``, ``mae_metric``: evaluation metrics.
    - ``total``, ``loss``: combined loss (``val_loss`` aliases ``val_total``).
    """
    model.eval()
    totals = {
        "total": 0.0,
        "reconstruction": 0.0,
        "mrstft": 0.0,
        "kl_raw": 0.0,
        "kl": 0.0,
        "kl_weighted": 0.0,
        "l1": 0.0,
        "nmse": 0.0,
        "nmse_metric": 0.0,
        "mae_metric": 0.0,
    }
    num_batches = 0

    alpha = float(cfg["loss"]["alpha_reconstruction"])
    gamma = float(cfg["loss"]["gamma_mrstft"])
    l1_w = float(cfg["loss"]["l1_weight"])
    nmse_w = float(cfg["loss"]["nmse_weight"])

    for batch in val_loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        outputs = model(inputs)
        recon = outputs["recon"]
        mu = outputs["mu"]
        logvar = outputs["logvar"]

        l1_term = l1_loss(recon, targets)
        nmse_term = nmse_loss(recon, targets)
        recon_term = l1_w * l1_term + nmse_w * nmse_term
        mrstft_term = mrstft_loss(recon, targets)
        # kl_raw: standard mean KL (no free bits), used for monitoring.
        kl_raw = kl_divergence_loss(mu, logvar, free_bits=0.0)
        # kl_term: KL with free bits (mirrors training backprop KL).
        kl_term = kl_divergence_loss(mu, logvar, free_bits=free_bits)
        total = alpha * recon_term + gamma * mrstft_term + beta_kl * kl_term
        totals["total"] += float(total.item())
        totals["reconstruction"] += float(recon_term.item())
        totals["mrstft"] += float(mrstft_term.item())
        totals["kl_raw"] += float(kl_raw.item())
        totals["kl"] += float(kl_term.item())
        totals["kl_weighted"] += beta_kl * float(kl_term.item())
        totals["l1"] += float(l1_term.item())
        totals["nmse"] += float(nmse_term.item())
        totals["nmse_metric"] += float(nmse(recon, targets))
        totals["mae_metric"] += float(mae(recon, targets))
        num_batches += 1

    averaged = {f"val_{k}": v / max(1, num_batches) for k, v in totals.items()}
    averaged["val_loss"] = averaged["val_total"]
    return averaged


def main() -> None:
    """CLI entrypoint for stage-3 VAE training."""
    parser = argparse.ArgumentParser(description="Train Stage-3 VAE WaveUNet")
    parser.add_argument("--config", default="configs/vae_stage3.yaml", help="Config file path")
    parser.add_argument("--dataset-file", default=None, help="Path to serialized dataset")
    parser.add_argument("--device", default=None, help="Device override (cuda/cpu)")
    parser.add_argument(
        "--warm-start-checkpoint",
        default=None,
        help="Optional checkpoint for warm-starting VAE from stage-1 weights",
    )
    args = parser.parse_args()

    cfg = normalize_config(load_config(args.config))
    device = resolve_device(args.device)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    dataset_file = resolve_existing_path(
        args.dataset_file or cfg.get("data", {}).get("dataset_file"),
        args.config,
    )
    if not dataset_file:
        raise ValueError("Stage-3 requires --dataset-file or data.dataset_file in config.")

    warm_start = resolve_existing_path(
        args.warm_start_checkpoint or cfg["training"].get("warm_start_checkpoint"),
        args.config,
    )

    print("Startup:")
    print(f"  Config path: {args.config}")
    print(f"  Dataset path: {dataset_file}")
    print(f"  Device: {device}")
    print(f"  Warm start checkpoint: {warm_start if warm_start else '<none>'}")

    info = get_dataset_info(dataset_file)
    print(
        f"  Dataset split type: {info.get('split_type', 'unknown')} | "
        f"frames={info.get('num_frames', 'unknown')}"
    )

    train_loader, val_loader = build_train_val_loaders(
        dataset_file=dataset_file,
        cfg=cfg,
        seed=seed,
    )

    model = create_model(cfg["model"], device=device)
    optimizer = build_optimizer(model, cfg)

    loaded_epoch: Optional[int] = None
    if warm_start:
        loaded_epoch = warm_start_from_checkpoint(model, warm_start, device)

    l1_loss = L1Loss()
    nmse_loss = NMSELoss()
    mrstft_loss = MultiResolutionSTFTLoss(
        fft_sizes=list(cfg["loss"]["mrstft_fft_sizes"]),
        hop_sizes=list(cfg["loss"]["mrstft_hop_sizes"]),
    )

    early_stopper = EarlyStopper(patience=int(cfg["training"]["early_stopping_patience"]))
    csv_logger = CSVLogger(str(cfg["logging"].get("csv_log"))) if cfg.get("logging", {}).get("csv_log") else None
    json_logger = JSONLogger(str(cfg["logging"].get("json_log"))) if cfg.get("logging", {}).get("json_log") else None

    checkpoint_dir = str(cfg["training"]["checkpoint_dir"])
    num_epochs = int(cfg["training"]["num_epochs"])

    # Resolve free bits once before training (0.0 = disabled = standard KL).
    free_bits_enabled = bool(cfg["kl_free_bits"].get("enabled", False))
    free_bits_value = float(cfg["kl_free_bits"].get("value", 0.0)) if free_bits_enabled else 0.0
    kl_schedule_type = str(cfg["kl_schedule"].get("type", "linear_warmup"))

    print(f"  KL schedule  : {kl_schedule_type}")
    print(f"  Free bits    : {'enabled (value=' + str(free_bits_value) + ')' if free_bits_enabled else 'disabled'}")

    # Save a config snapshot for reproducibility.
    exp_name_snap = str(cfg.get("experiment_name", "vae_stage3"))
    snapshot_dir = Path("logs") / exp_name_snap
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / "config_snapshot.yaml"
    with open(snapshot_path, "w", encoding="utf-8") as _snap:
        _snap.write(f"# source_config: {args.config}\n")
        yaml.safe_dump(cfg, _snap, sort_keys=False)

    start = time.time()
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        beta_kl = compute_beta_kl(cfg, epoch, num_epochs)

        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            l1_loss=l1_loss,
            nmse_loss=nmse_loss,
            mrstft_loss=mrstft_loss,
            cfg=cfg,
            beta_kl=beta_kl,
            free_bits=free_bits_value,
        )
        val_metrics = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            device=device,
            l1_loss=l1_loss,
            nmse_loss=nmse_loss,
            mrstft_loss=mrstft_loss,
            cfg=cfg,
            beta_kl=beta_kl,
            free_bits=free_bits_value,
        )

        log_row: Dict[str, Any] = {
            "epoch": epoch,
            "beta_kl": beta_kl,
            "free_bits_enabled": free_bits_enabled,
            **train_metrics,
            **val_metrics,
            "time": time.time() - epoch_start,
        }
        if csv_logger is not None:
            csv_logger.log(log_row)
        if json_logger is not None:
            json_logger.log(log_row)

        Checkpoint.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=log_row,
            filepath=f"{checkpoint_dir}/latest_stage3.pth",
            scheduler=None,
        )

        if val_metrics["val_total"] <= early_stopper.best_loss:
            Checkpoint.save(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=log_row,
                filepath=f"{checkpoint_dir}/best_stage3.pth",
                scheduler=None,
            )

        print(
            f"Epoch {epoch}/{num_epochs} "
            f"train_total={train_metrics['train_total']:.6f} "
            f"recon={train_metrics['train_reconstruction']:.6f} "
            f"kl_raw={train_metrics['train_kl_raw']:.6f} "
            f"kl={train_metrics['train_kl']:.6f} "
            f"kl_w={train_metrics['train_kl_weighted']:.6g} "
            f"| val_total={val_metrics['val_total']:.6f} "
            f"val_kl_raw={val_metrics['val_kl_raw']:.6f} "
            f"val_kl_w={val_metrics['val_kl_weighted']:.6g} "
            f"beta_kl={beta_kl:.6g}"
        )

        if early_stopper.step(val_metrics["val_total"], epoch):
            print(f"Early stopping at epoch {epoch}")
            break

    runtime = time.time() - start
    experiment_name = str(cfg.get("experiment_name", "vae_stage3"))
    summary_path = Path("logs") / experiment_name / "run_summary.json"
    summary_payload = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config_path": args.config,
        "config_snapshot": str(snapshot_path),
        "dataset_path": dataset_file,
        "device": device,
        "warm_start_checkpoint": warm_start,
        "loaded_checkpoint_epoch": loaded_epoch,
        "kl_schedule": cfg.get("kl_schedule", {}),
        "kl_free_bits_enabled": free_bits_enabled,
        "kl_free_bits_value": free_bits_value,
        "checkpoint_best": f"{checkpoint_dir}/best_stage3.pth",
        "checkpoint_latest": f"{checkpoint_dir}/latest_stage3.pth",
        "csv_log": cfg["logging"].get("csv_log"),
        "json_log": cfg["logging"].get("json_log"),
        "runtime_sec": runtime,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, sort_keys=True)

    latest_pointer = Path("logs") / "LATEST_STAGE3_VAE.json"
    with open(latest_pointer, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "created_at": summary_payload["created_at"],
                "experiment_name": experiment_name,
                "run_summary": str(summary_path),
                "csv_log": summary_payload["csv_log"],
                "json_log": summary_payload["json_log"],
                "checkpoint_best": summary_payload["checkpoint_best"],
                "checkpoint_latest": summary_payload["checkpoint_latest"],
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    print(f"Run summary written: {summary_path}")
    print(f"Latest stage-3 pointer: {latest_pointer}")


if __name__ == "__main__":
    main()
