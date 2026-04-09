#!/usr/bin/env python3
"""Train B1 branch: fixed band-split + tiny neural gain controller."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.serialized import get_dataset_info, load_serialized_dataset
from src.dsp.bandsplit import FixedBandSplitter, peak_safe
from src.evaluation.evaluate import evaluate_batch
from src.losses.reconstruction import L1Loss, NMSELoss
from src.models.band_controller import BandController
from src.training.callbacks import EarlyStopper
from src.training.checkpointing import Checkpoint
from src.utils.logging import CSVLogger, JSONLogger
from src.utils.seed import set_seed


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(config)
    cfg.setdefault("experiment_name", "band_controller")
    cfg.setdefault("seed", 42)
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("dsp", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("training", {})

    cfg["data"].setdefault("dataset_file", "")
    cfg["data"].setdefault("sample_rate", 44100)

    cfg["dsp"].setdefault("bands_hz", [[20, 180], [180, 1200], [1200, 6000], [6000, 22050]])
    cfg["dsp"].setdefault("peak_limit", 0.99)

    cfg["model"].setdefault("hidden_channels", 16)
    cfg["model"].setdefault("max_gain_db", 6.0)
    cfg["model"].setdefault("min_gain_db", -6.0)

    cfg["loss"].setdefault("l1_weight", 1.0)
    cfg["loss"].setdefault("nmse_weight", 1.0)

    cfg["training"].setdefault("split_train", "train")
    cfg["training"].setdefault("split_val", "val")
    cfg["training"].setdefault("batch_size", 16)
    cfg["training"].setdefault("num_workers", 2)
    cfg["training"].setdefault("num_epochs", 30)
    cfg["training"].setdefault("max_train_batches", 0)
    cfg["training"].setdefault("max_val_batches", 0)
    cfg["training"].setdefault("lr", 1e-4)
    cfg["training"].setdefault("weight_decay", 1e-5)
    cfg["training"].setdefault("gradient_clip", 1.0)
    cfg["training"].setdefault("early_stopping_patience", 8)
    cfg["training"].setdefault("device", "cpu")
    cfg["training"].setdefault("output_root", "outputs/band_controller")
    return cfg


def resolve_device(device: str) -> str:
    d = device.lower()
    if d == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    if d not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device '{device}'")
    return d


def recon_loss(out: torch.Tensor, tgt: torch.Tensor, l1: L1Loss, nmse: NMSELoss, cfg: Dict[str, Any]) -> torch.Tensor:
    return float(cfg["loss"]["l1_weight"]) * l1(out, tgt) + float(cfg["loss"]["nmse_weight"]) * nmse(out, tgt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train band-split tiny controller")
    parser.add_argument("--config", default="configs/band_controller.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dataset-file", default=None)
    args = parser.parse_args()

    cfg = normalize_config(load_config(args.config))
    set_seed(int(cfg["seed"]))

    dataset_file = args.dataset_file or cfg["data"]["dataset_file"]
    if not dataset_file:
        raise ValueError("Dataset path missing. Set data.dataset_file or --dataset-file")

    device = resolve_device(args.device or str(cfg["training"]["device"]))
    info = get_dataset_info(dataset_file)
    print(f"Dataset split type={info.get('split_type', 'unknown')} frames={info.get('num_frames', 'unknown')}")

    train_loader = load_serialized_dataset(
        dataset_file,
        split=cfg["training"]["split_train"],
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"]["num_workers"]),
        validate=False,
    )
    val_loader = load_serialized_dataset(
        dataset_file,
        split=cfg["training"]["split_val"],
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"]["num_workers"]),
        validate=False,
    )

    splitter = FixedBandSplitter(sample_rate=int(cfg["data"]["sample_rate"]), bands_hz=cfg["dsp"]["bands_hz"])
    model = BandController(
        num_bands=len(cfg["dsp"]["bands_hz"]),
        hidden_channels=int(cfg["model"]["hidden_channels"]),
        max_gain_db=float(cfg["model"]["max_gain_db"]),
        min_gain_db=float(cfg["model"]["min_gain_db"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    l1 = L1Loss()
    nmse = NMSELoss()
    stopper = EarlyStopper(patience=int(cfg["training"]["early_stopping_patience"]))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{cfg['experiment_name']}_{ts}"
    run_dir = Path(cfg["training"]["output_root"]) / run_name
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        f.write(f"# source_config: {args.config}\n")
        yaml.safe_dump(cfg, f, sort_keys=False)

    csv_logger = CSVLogger(str(run_dir / "metrics.csv"))
    json_logger = JSONLogger(str(run_dir / "metrics.jsonl"))

    max_train_batches = int(cfg["training"].get("max_train_batches", 0))
    max_val_batches = int(cfg["training"].get("max_val_batches", 0))

    for epoch in range(1, int(cfg["training"]["num_epochs"]) + 1):
        model.train()
        train_loss, train_nmse, n_train = 0.0, 0.0, 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)

            bands = splitter.analyze(inp)
            gains = model(inp)
            out = splitter.synthesize(splitter.apply_band_gains(bands, gains))
            out = peak_safe(out, peak_limit=float(cfg["dsp"]["peak_limit"]))

            loss = recon_loss(out, tgt, l1, nmse, cfg)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gc = float(cfg["training"]["gradient_clip"])
            if gc > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gc)
            optimizer.step()

            m = evaluate_batch(out.detach(), tgt.detach())
            train_loss += float(loss.item())
            train_nmse += float(m["batch_nmse"])
            n_train += 1

            if max_train_batches > 0 and batch_idx >= max_train_batches:
                break

        model.eval()
        val_loss, val_nmse, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader, start=1):
                inp = batch["input"].to(device)
                tgt = batch["target"].to(device)
                bands = splitter.analyze(inp)
                gains = model(inp)
                out = splitter.synthesize(splitter.apply_band_gains(bands, gains))
                out = peak_safe(out, peak_limit=float(cfg["dsp"]["peak_limit"]))
                loss = recon_loss(out, tgt, l1, nmse, cfg)
                m = evaluate_batch(out, tgt)
                val_loss += float(loss.item())
                val_nmse += float(m["batch_nmse"])
                n_val += 1

                if max_val_batches > 0 and batch_idx >= max_val_batches:
                    break

        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(n_train, 1),
            "train_nmse": train_nmse / max(n_train, 1),
            "val_loss": val_loss / max(n_val, 1),
            "val_nmse": val_nmse / max(n_val, 1),
        }
        csv_logger.log(row)
        json_logger.log(row)

        Checkpoint.save(model, optimizer, epoch, row, str(ckpt_dir / "latest.pth"), scheduler=None)
        if row["val_loss"] <= stopper.best_loss:
            Checkpoint.save(model, optimizer, epoch, row, str(ckpt_dir / "best.pth"), scheduler=None)

        print(
            f"Epoch {epoch}: train_loss={row['train_loss']:.6f} val_loss={row['val_loss']:.6f} "
            f"train_nmse={row['train_nmse']:.6f} val_nmse={row['val_nmse']:.6f}"
        )

        if stopper.step(row["val_loss"], epoch):
            print(f"Early stopping at epoch {epoch}")
            break

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "config_path": args.config,
        "dataset_file": dataset_file,
        "device": device,
        "checkpoint_best": str(ckpt_dir / "best.pth"),
        "checkpoint_latest": str(ckpt_dir / "latest.pth"),
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(Path(cfg["training"]["output_root"]) / "LATEST_BAND_CONTROLLER.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
