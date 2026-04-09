#!/usr/bin/env python3
"""Run fast deterministic DSP baseline on serialized dataset splits."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.serialized import get_dataset_info, load_serialized_dataset
from src.dsp.fast_baseline import FastDSPBaseline
from src.evaluation.evaluate import evaluate_batch
from src.utils.seed import set_seed


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize DSP baseline config with safe defaults."""
    cfg = dict(config)
    cfg.setdefault("experiment_name", "dsp_fast_baseline")
    cfg.setdefault("seed", 42)
    cfg.setdefault("data", {})
    cfg.setdefault("dsp", {})
    cfg.setdefault("run", {})

    data_cfg = cfg["data"]
    data_cfg.setdefault("dataset_file", "")

    dsp_cfg = cfg["dsp"]
    dsp_cfg.setdefault("target_level_db", -14.0)
    dsp_cfg.setdefault("max_gain_db", 12.0)
    dsp_cfg.setdefault("min_gain_db", -12.0)
    dsp_cfg.setdefault("peak_limit", 0.99)
    dsp_cfg.setdefault("smoothing", {"enabled": True, "alpha": 0.9})
    dsp_cfg.setdefault(
        "compression",
        {"enabled": False, "threshold_db": -6.0, "ratio": 2.0},
    )

    run_cfg = cfg["run"]
    run_cfg.setdefault("split", "val")
    run_cfg.setdefault("batch_size", 64)
    run_cfg.setdefault("num_workers", 2)
    run_cfg.setdefault("device", "cpu")
    run_cfg.setdefault("output_root", "outputs/dsp_baseline")
    run_cfg.setdefault("save_outputs", True)
    run_cfg.setdefault("save_limit_frames", 5000)

    return cfg


def resolve_device(requested_device: str) -> str:
    """Resolve runtime device with safe CUDA fallback."""
    device = requested_device.lower()
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device '{requested_device}'.")
    return device


def _avg_dicts(items: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute arithmetic mean over list of metric dictionaries."""
    if not items:
        return {}
    totals: Dict[str, float] = {}
    for row in items:
        for key, value in row.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return {k: v / len(items) for k, v in totals.items()}


def _prefix_keys(values: Dict[str, float], prefix: str) -> Dict[str, float]:
    """Prefix dictionary keys to keep metrics unambiguous."""
    return {f"{prefix}_{k}": float(v) for k, v in values.items()}


def main() -> None:
    """CLI entrypoint for deterministic DSP baseline execution."""
    parser = argparse.ArgumentParser(description="Run fast deterministic DSP baseline")
    parser.add_argument("--config", default="configs/dsp_baseline.yaml", help="Path to config YAML")
    parser.add_argument("--dataset-file", default=None, help="Optional dataset override")
    parser.add_argument("--split", default=None, choices=["train", "val", "test"], help="Split override")
    parser.add_argument("--device", default=None, help="Device override (cpu/cuda)")
    args = parser.parse_args()

    cfg = normalize_config(load_config(args.config))
    set_seed(int(cfg.get("seed", 42)))

    dataset_file = args.dataset_file or cfg["data"].get("dataset_file")
    if not dataset_file:
        raise ValueError("Dataset path missing. Set data.dataset_file or --dataset-file.")

    split = args.split or str(cfg["run"].get("split", "val"))
    run_device = resolve_device(args.device or str(cfg["run"].get("device", "cpu")))

    info = get_dataset_info(dataset_file)
    print("Startup:")
    print(f"  Config path: {args.config}")
    print(f"  Dataset path: {dataset_file}")
    print(f"  Split: {split}")
    print(f"  Device: {run_device}")
    print(
        f"  Dataset split type: {info.get('split_type', 'unknown')} | "
        f"frames={info.get('num_frames', 'unknown')}"
    )

    dataloader = load_serialized_dataset(
        dataset_file,
        split=split,
        batch_size=int(cfg["run"]["batch_size"]),
        num_workers=int(cfg["run"]["num_workers"]),
        validate=False,
    )

    processor = FastDSPBaseline(cfg["dsp"], device=run_device)
    processor.reset()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{cfg['experiment_name']}_{split}_{timestamp}"
    run_dir = Path(cfg["run"]["output_root"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = run_dir / "config_snapshot.yaml"
    with open(snapshot_path, "w", encoding="utf-8") as handle:
        handle.write(f"# source_config: {args.config}\n")
        yaml.safe_dump(cfg, handle, sort_keys=False)

    metric_rows_output: List[Dict[str, float]] = []
    metric_rows_input: List[Dict[str, float]] = []
    dsp_rows: List[Dict[str, float]] = []

    save_outputs = bool(cfg["run"].get("save_outputs", True))
    save_limit_frames = int(cfg["run"].get("save_limit_frames", 5000))

    cached_inputs: List[np.ndarray] = []
    cached_outputs: List[np.ndarray] = []
    cached_targets: List[np.ndarray] = []
    cached_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader, start=1):
            inputs = batch["input"].to(run_device)
            targets = batch["target"].to(run_device)

            outputs, dsp_stats = processor.process_batch(inputs)

            metric_rows_output.append(evaluate_batch(outputs, targets))
            metric_rows_input.append(evaluate_batch(inputs, targets))
            dsp_rows.append(dsp_stats)

            if save_outputs and cached_count < save_limit_frames:
                remaining = save_limit_frames - cached_count
                keep = min(remaining, inputs.shape[0])
                cached_inputs.append(inputs[:keep].cpu().numpy())
                cached_outputs.append(outputs[:keep].cpu().numpy())
                cached_targets.append(targets[:keep].cpu().numpy())
                cached_count += keep

            if batch_idx % 20 == 0:
                print(f"  Processed {batch_idx} batches...")

    output_metrics = _prefix_keys(_avg_dicts(metric_rows_output), "output")
    input_metrics = _prefix_keys(_avg_dicts(metric_rows_input), "input")
    dsp_metrics = _prefix_keys(_avg_dicts(dsp_rows), "dsp")

    summary: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment_name": cfg["experiment_name"],
        "run_name": run_name,
        "config_path": args.config,
        "config_snapshot": str(snapshot_path),
        "dataset_path": dataset_file,
        "split": split,
        "device": run_device,
        "num_batches": len(metric_rows_output),
        "saved_frames": cached_count,
        **input_metrics,
        **output_metrics,
        **dsp_metrics,
    }

    if save_outputs and cached_count > 0:
        npz_path = run_dir / f"outputs_{split}.npz"
        np.savez_compressed(
            npz_path,
            inputs=np.concatenate(cached_inputs, axis=0),
            outputs=np.concatenate(cached_outputs, axis=0),
            targets=np.concatenate(cached_targets, axis=0),
        )
        summary["outputs_file"] = str(npz_path)

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    # Keep a compact single-row CSV for easy comparison across runs.
    csv_path = run_dir / "metrics.csv"
    metric_keys = sorted(summary.keys())
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write(",".join(metric_keys) + "\n")
        handle.write(",".join(str(summary[k]) for k in metric_keys) + "\n")

    latest_pointer = Path(cfg["run"]["output_root"]) / "LATEST_DSP_BASELINE.json"
    with open(latest_pointer, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "created_at": summary["created_at"],
                "run_name": run_name,
                "summary": str(summary_path),
                "metrics_csv": str(csv_path),
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    print("Run complete:")
    print(f"  Run directory: {run_dir}")
    print(f"  Summary: {summary_path}")
    print(f"  Metrics CSV: {csv_path}")
    if save_outputs and cached_count > 0:
        print(f"  Saved outputs: {summary.get('outputs_file')}")
    print(f"  Latest pointer: {latest_pointer}")


if __name__ == "__main__":
    main()
