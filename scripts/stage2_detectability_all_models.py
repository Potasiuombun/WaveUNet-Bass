#!/usr/bin/env python3
"""Stage-2 detectability fine-tuning for all trained Stage-1 models.

This script fine-tunes all Stage-1 model families using detectability between
normalized input and model output as the primary loss.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
import yaml

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.serialized import load_serialized_dataset
from src.dsp.bandsplit import FixedBandSplitter, peak_safe
from src.dsp.ddsp_controls import apply_ddsp_controls
from src.dsp.fast_baseline import FastDSPBaseline
from src.losses.perceptual import DetectabilityLossWrapper
from src.losses.reconstruction import L1Loss, NMSELoss
from src.models.band_controller import BandController
from src.models.ddsp_controller import DDSPController
from src.models.factory import create_model
from src.models.tiny_residual import TinyResidualModel, apply_residual_with_peak_safety
from src.training.callbacks import EarlyStopper
from src.training.checkpointing import Checkpoint
from src.training.metrics import mae, nmse
from src.utils.seed import set_seed


def resolve_device(requested: str) -> str:
    device = requested.lower()
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device '{requested}'")
    return device


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint_state(checkpoint_path: Path, device: str) -> Dict[str, Any]:
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict):
        if "model_state_dict" in state:
            return state["model_state_dict"]
        if "model" in state:
            return state["model"]
        if "state_dict" in state:
            return state["state_dict"]
    if isinstance(state, dict):
        return state
    raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")


def build_model_bundle(root: Path, model_name: str, device: str) -> Dict[str, Any]:
    """Build model and any DSP helpers required for forward synthesis."""
    if model_name == "waveunet":
        cfg = load_yaml(root / "configs/stage1_fma_curated_cli_norm.yaml")
        model = create_model(cfg["model"], device=device)
        ckpt = root / "checkpoints/stage1_fma_curated_cli_norm/best.pth"
        model.load_state_dict(load_checkpoint_state(ckpt, device))
        return {"model": model, "checkpoint": str(ckpt), "type": model_name}

    if model_name == "vae_waveunet":
        cfg = load_yaml(root / "configs/stage1_vae_fma_curated_cli_norm.yaml")
        model = create_model(cfg["model"], device=device)
        ckpt = root / "checkpoints/stage1_vae_fma_curated_cli_norm/best_stage3.pth"
        model.load_state_dict(load_checkpoint_state(ckpt, device))
        return {"model": model, "checkpoint": str(ckpt), "type": model_name}

    if model_name == "dsp_tiny_residual":
        cfg = load_yaml(root / "configs/stage1_dsp_residual_fma_curated_cli_norm.yaml")
        model = TinyResidualModel(
            hidden_channels=int(cfg["model"]["hidden_channels"]),
            num_layers=int(cfg["model"]["num_layers"]),
            kernel_size=int(cfg["model"]["kernel_size"]),
            activation=str(cfg["model"]["activation"]),
            max_residual=float(cfg["model"]["max_residual"]),
        ).to(device)
        latest = load_yaml(root / "outputs/stage1_dsp_residual/LATEST_DSP_RESIDUAL.json")
        ckpt = root / latest["checkpoint_best"]
        model.load_state_dict(load_checkpoint_state(ckpt, device))
        dsp = FastDSPBaseline(cfg["dsp"], device=device)
        return {
            "model": model,
            "checkpoint": str(ckpt),
            "type": model_name,
            "dsp": dsp,
            "peak_limit": float(cfg["dsp"].get("peak_limit", 0.99)),
        }

    if model_name == "band_controller":
        cfg = load_yaml(root / "configs/stage1_band_controller_fma_curated_cli_norm.yaml")
        model = BandController(
            num_bands=len(cfg["dsp"]["bands_hz"]),
            hidden_channels=int(cfg["model"]["hidden_channels"]),
            max_gain_db=float(cfg["model"]["max_gain_db"]),
            min_gain_db=float(cfg["model"]["min_gain_db"]),
        ).to(device)
        latest = load_yaml(root / "outputs/stage1_band_controller/LATEST_BAND_CONTROLLER.json")
        ckpt = root / latest["checkpoint_best"]
        model.load_state_dict(load_checkpoint_state(ckpt, device))
        splitter = FixedBandSplitter(sample_rate=int(cfg["data"]["sample_rate"]), bands_hz=cfg["dsp"]["bands_hz"])
        return {
            "model": model,
            "checkpoint": str(ckpt),
            "type": model_name,
            "splitter": splitter,
            "peak_limit": float(cfg["dsp"].get("peak_limit", 0.99)),
        }

    if model_name == "ddsp_controller":
        cfg = load_yaml(root / "configs/stage1_ddsp_controller_fma_curated_cli_norm.yaml")
        model = DDSPController(
            num_bands=len(cfg["dsp"]["bands_hz"]),
            hidden_channels=int(cfg["model"]["hidden_channels"]),
            max_gain_db=float(cfg["model"]["max_gain_db"]),
            min_gain_db=float(cfg["model"]["min_gain_db"]),
            max_tilt_db=float(cfg["model"]["max_tilt_db"]),
            envelope_enabled=bool(cfg["model"]["envelope_enabled"]),
            min_envelope=float(cfg["model"]["min_envelope"]),
            max_envelope=float(cfg["model"]["max_envelope"]),
        ).to(device)
        latest = load_yaml(root / "outputs/stage1_ddsp_controller/LATEST_DDSP_CONTROLLER.json")
        ckpt = root / latest["checkpoint_best"]
        model.load_state_dict(load_checkpoint_state(ckpt, device))
        splitter = FixedBandSplitter(sample_rate=int(cfg["data"]["sample_rate"]), bands_hz=cfg["dsp"]["bands_hz"])
        return {
            "model": model,
            "checkpoint": str(ckpt),
            "type": model_name,
            "splitter": splitter,
            "peak_limit": float(cfg["dsp"].get("peak_limit", 0.99)),
        }

    raise ValueError(f"Unknown model name '{model_name}'")


def model_forward(bundle: Dict[str, Any], x: torch.Tensor) -> torch.Tensor:
    model_type = bundle["type"]
    model = bundle["model"]

    if model_type == "waveunet":
        return model(x)

    if model_type == "vae_waveunet":
        out = model(x)
        return out["recon"]

    if model_type == "dsp_tiny_residual":
        dsp: FastDSPBaseline = bundle["dsp"]
        dsp_out, _ = dsp.process_batch(x)
        residual = model(dsp_out)
        return apply_residual_with_peak_safety(dsp_out, residual, peak_limit=float(bundle["peak_limit"]))

    if model_type == "band_controller":
        splitter: FixedBandSplitter = bundle["splitter"]
        bands = splitter.analyze(x)
        gains = model(x)
        y = splitter.synthesize(splitter.apply_band_gains(bands, gains))
        return peak_safe(y, peak_limit=float(bundle["peak_limit"]))

    if model_type == "ddsp_controller":
        splitter: FixedBandSplitter = bundle["splitter"]
        controls = model(x)
        return apply_ddsp_controls(x, splitter, controls, peak_limit=float(bundle["peak_limit"]))

    raise ValueError(f"Unhandled model type '{model_type}'")


def run_one_model(
    root: Path,
    model_name: str,
    dataset_file: Path,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    early_patience: int,
    detectability_weight: float,
    recon_weight: float,
) -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print(f"Stage-2 Detectability Fine-tuning: {model_name}")
    print("=" * 90)

    bundle = build_model_bundle(root, model_name, device)
    model: torch.nn.Module = bundle["model"]
    model.train()

    train_loader = load_serialized_dataset(
        str(dataset_file),
        split="train",
        batch_size=batch_size,
        num_workers=2,
        validate=False,
    )
    val_loader = load_serialized_dataset(
        str(dataset_file),
        split="val",
        batch_size=batch_size,
        num_workers=2,
        validate=False,
    )

    detect_loss = DetectabilityLossWrapper(enabled=True, sample_rate=44100, frame_size=1024)
    l1_loss = L1Loss()
    nmse_loss = NMSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    stopper = EarlyStopper(patience=early_patience)

    save_dir = root / f"checkpoints/stage2_detectability_{model_name}"
    log_dir = root / f"logs/stage2_detectability_{model_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[Dict[str, float]] = []
    best_val = float("inf")
    best_epoch = 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        if bundle["type"] == "dsp_tiny_residual":
            bundle["dsp"].reset()

        train_tot = 0.0
        train_det = 0.0
        train_rec = 0.0
        train_nmse = 0.0
        n_train = 0

        for batch in train_loader:
            x = batch["input"].to(device)
            # Stage-2 objective requested: compare normalized input vs output detectability.
            y_ref = x

            optimizer.zero_grad(set_to_none=True)
            y_hat = model_forward(bundle, x)

            det = detect_loss(y_hat, y_ref)
            rec = l1_loss(y_hat, y_ref) + nmse_loss(y_hat, y_ref)
            total = detectability_weight * det + recon_weight * rec
            total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            bs = x.shape[0]
            train_tot += float(total.item()) * bs
            train_det += float(det.item()) * bs
            train_rec += float(rec.item()) * bs
            train_nmse += nmse(y_hat, y_ref) * bs
            n_train += bs

        model.eval()
        if bundle["type"] == "dsp_tiny_residual":
            bundle["dsp"].reset()

        val_tot = 0.0
        val_det = 0.0
        val_rec = 0.0
        val_nmse = 0.0
        val_mae = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["input"].to(device)
                y_ref = x

                y_hat = model_forward(bundle, x)
                det = detect_loss(y_hat, y_ref)
                rec = l1_loss(y_hat, y_ref) + nmse_loss(y_hat, y_ref)
                total = detectability_weight * det + recon_weight * rec

                bs = x.shape[0]
                val_tot += float(total.item()) * bs
                val_det += float(det.item()) * bs
                val_rec += float(rec.item()) * bs
                val_nmse += nmse(y_hat, y_ref) * bs
                val_mae += mae(y_hat, y_ref) * bs
                n_val += bs

        row = {
            "epoch": float(epoch),
            "train_total": train_tot / max(n_train, 1),
            "train_detectability": train_det / max(n_train, 1),
            "train_recon": train_rec / max(n_train, 1),
            "train_nmse": train_nmse / max(n_train, 1),
            "val_total": val_tot / max(n_val, 1),
            "val_detectability": val_det / max(n_val, 1),
            "val_recon": val_rec / max(n_val, 1),
            "val_nmse": val_nmse / max(n_val, 1),
            "val_mae": val_mae / max(n_val, 1),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_sec": float(time.time() - start),
        }
        metrics_rows.append(row)

        Checkpoint.save(model, optimizer, epoch, row, str(save_dir / "latest_stage2.pth"), scheduler=None)
        if row["val_total"] < best_val:
            best_val = row["val_total"]
            best_epoch = epoch
            Checkpoint.save(model, optimizer, epoch, row, str(save_dir / "best_stage2.pth"), scheduler=None)

        print(
            f"Epoch {epoch}/{epochs} "
            f"train_det={row['train_detectability']:.6f} val_det={row['val_detectability']:.6f} "
            f"val_total={row['val_total']:.6f} val_nmse={row['val_nmse']:.6f}"
        )

        if stopper.step(row["val_total"], epoch):
            print(f"Early stopping at epoch {epoch}")
            break

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = log_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    summary = {
        "model": model_name,
        "stage1_checkpoint": bundle["checkpoint"],
        "stage2_best_checkpoint": str(save_dir / "best_stage2.pth"),
        "stage2_latest_checkpoint": str(save_dir / "latest_stage2.pth"),
        "metrics_csv": str(metrics_path),
        "best_epoch": int(best_epoch),
        "best_val_total": float(best_val),
        "last_val_detectability": float(metrics_rows[-1]["val_detectability"]),
        "last_val_nmse": float(metrics_rows[-1]["val_nmse"]),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    with open(log_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    with open(root / f"logs/LATEST_STAGE2_{model_name.upper()}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage-2 detectability fine-tuning for all models")
    parser.add_argument(
        "--dataset-file",
        default="datasets/stage1_fma_curated_cli_norm.pkl",
        help="Serialized dataset file",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-patience", type=int, default=3)
    parser.add_argument(
        "--detectability-weight",
        type=float,
        default=1.0,
        help="Weight for detectability(input_norm, output)",
    )
    parser.add_argument(
        "--recon-weight",
        type=float,
        default=0.05,
        help="Small stabilizer weight for (L1 + NMSE). Set 0.0 for pure detectability.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    dataset_file = (root / args.dataset_file).resolve()
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")

    set_seed(42)
    device = resolve_device(args.device)

    models = [
        "waveunet",
        "vae_waveunet",
        "dsp_tiny_residual",
        "band_controller",
        "ddsp_controller",
    ]

    all_summaries: List[Dict[str, Any]] = []
    for model_name in models:
        summary = run_one_model(
            root=root,
            model_name=model_name,
            dataset_file=dataset_file,
            device=device,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            grad_clip=float(args.grad_clip),
            early_patience=int(args.early_patience),
            detectability_weight=float(args.detectability_weight),
            recon_weight=float(args.recon_weight),
        )
        all_summaries.append(summary)

    leaderboard = pd.DataFrame(
        [
            {
                "model": s["model"],
                "best_val_total": s["best_val_total"],
                "last_val_detectability": s["last_val_detectability"],
                "last_val_nmse": s["last_val_nmse"],
                "best_epoch": s["best_epoch"],
                "stage2_best_checkpoint": s["stage2_best_checkpoint"],
            }
            for s in all_summaries
        ]
    ).sort_values("best_val_total", ascending=True)

    out_csv = root / "logs/stage2_detectability_all_models_leaderboard.csv"
    out_json = root / "logs/stage2_detectability_all_models_summary.json"
    leaderboard.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, sort_keys=True)

    print("\n" + "=" * 90)
    print("Stage-2 Detectability Summary (lower is better)")
    print("=" * 90)
    for i, row in enumerate(leaderboard.to_dict(orient="records"), start=1):
        print(
            f"{i}. {row['model']}: best_val_total={row['best_val_total']:.6f}, "
            f"val_detectability={row['last_val_detectability']:.6f}, val_nmse={row['last_val_nmse']:.6f}"
        )
    print(f"\nSaved leaderboard: {out_csv}")
    print(f"Saved summary: {out_json}")


if __name__ == "__main__":
    main()
