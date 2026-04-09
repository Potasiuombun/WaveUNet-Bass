#!/usr/bin/env python3
"""Compare current baselines against ADMM outputs on a small track subset.

This script evaluates methods on tracks from a directory containing:
- <track_id>_reference_clipped.npy
- <track_id>_admm_processed.npy

Methods compared:
- input (reference_clipped as-is)
- admm (oracle baseline target itself)
- dsp_fast (FastDSPBaseline)
- waveunet_stage1 (Residual WaveUNet checkpoint)
- vae_plain_cyclical (plain VAE checkpoint)

Metrics are computed against ADMM output to provide quick alignment checks.
"""

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dsp.fast_baseline import FastDSPBaseline
from src.models.factory import create_model
from src.training.metrics import mae, nmse
from src.utils.seed import set_seed

try:
    from libdetectability import DetectabilityLoss  # type: ignore
except Exception:
    DetectabilityLoss = None  # type: ignore[assignment]


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_model_from_checkpoint(
    model_cfg: Dict[str, Any], checkpoint_path: str, device: str
) -> torch.nn.Module:
    """Instantiate model from config and load checkpoint weights."""
    model = create_model(model_cfg, device=device)
    state = torch.load(checkpoint_path, map_location=device)
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(model_state, strict=False)
    model.eval()
    return model


def to_tensor_1ct(wave: np.ndarray, device: str) -> torch.Tensor:
    """Convert numpy waveform to torch tensor shape [1, 1, T]."""
    arr = np.asarray(wave, dtype=np.float32).reshape(-1)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def tensor_to_numpy_1d(x: torch.Tensor) -> np.ndarray:
    """Convert tensor shape [1, 1, T] (or compatible) back to numpy [T]."""
    return x.detach().cpu().reshape(-1).numpy()


def peak(x: np.ndarray) -> float:
    """Compute max absolute sample value."""
    return float(np.max(np.abs(x)))


def rms_db(x: np.ndarray, eps: float = 1e-10) -> float:
    """Compute RMS level in dBFS."""
    rms = np.sqrt(np.mean(np.square(x)) + eps)
    return float(20.0 * np.log10(max(rms, eps)))


def compute_metrics(output: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute scalar metrics against target waveform."""
    out_t = torch.from_numpy(output.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    tgt_t = torch.from_numpy(target.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return {
        "nmse_vs_admm": float(nmse(out_t, tgt_t)),
        "mae_vs_admm": float(mae(out_t, tgt_t)),
        "output_peak": peak(output),
        "output_rms_db": rms_db(output),
    }


def compute_detectability(
    output: np.ndarray,
    target: np.ndarray,
    detector: Optional[Any],
    frame_size: int,
) -> Optional[float]:
    """Compute detectability loss if detector is available."""
    if detector is None:
        return None

    n = min(len(output), len(target))
    usable = (n // frame_size) * frame_size
    if usable < frame_size:
        return None

    out_trim = output[:usable].astype(np.float32).reshape(-1, frame_size)
    tgt_trim = target[:usable].astype(np.float32).reshape(-1, frame_size)
    out_t = torch.from_numpy(out_trim)
    tgt_t = torch.from_numpy(tgt_trim)
    with torch.no_grad():
        return float(detector(out_t, tgt_t).item())


def discover_tracks(admm_dir: Path) -> List[str]:
    """Return sorted track IDs discovered from *_admm_processed.npy files."""
    ids = []
    for path in sorted(admm_dir.glob("*_admm_processed.npy")):
        ids.append(path.name.replace("_admm_processed.npy", ""))
    return ids


def run() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Compare DSP/ML outputs against ADMM baselines")
    parser.add_argument("--admm-dir", default="/home/tudor/Documents/MasterThesis/018_output")
    parser.add_argument("--num-tracks", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--waveunet-config", default="configs/baseline_stage1.yaml")
    parser.add_argument("--waveunet-checkpoint", default="checkpoints/baseline_stage1/best.pth")
    parser.add_argument("--vae-config", default="configs/vae_plain_cyclical.yaml")
    parser.add_argument("--vae-checkpoint", default="checkpoints/vae_plain_cyclical/best_stage3.pth")
    parser.add_argument("--dsp-config", default="configs/dsp_baseline.yaml")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--frame-size", type=int, default=1024)
    parser.add_argument("--detectability", action="store_true", help="Enable libdetectability metric")
    parser.add_argument("--peaq", action="store_true", help="Attempt PEAQ metric (if external tool exists)")
    args = parser.parse_args()

    set_seed(args.seed)
    admm_dir = Path(args.admm_dir)
    if not admm_dir.exists():
        raise FileNotFoundError(f"ADMM directory not found: {admm_dir}")

    all_tracks = discover_tracks(admm_dir)
    if not all_tracks:
        raise ValueError(f"No *_admm_processed.npy files found in {admm_dir}")

    tracks = all_tracks[: max(1, args.num_tracks)]

    print("Loading models/processors...")
    wave_cfg = load_yaml(args.waveunet_config)
    vae_cfg = load_yaml(args.vae_config)
    dsp_cfg = load_yaml(args.dsp_config)

    waveunet = load_model_from_checkpoint(wave_cfg["model"], args.waveunet_checkpoint, args.device)
    vae = load_model_from_checkpoint(vae_cfg["model"], args.vae_checkpoint, args.device)
    dsp = FastDSPBaseline(dsp_cfg["dsp"], device=args.device)
    dsp.reset()

    detector = None
    detectability_available = DetectabilityLoss is not None
    if args.detectability:
        if not detectability_available:
            print("Detectability requested but libdetectability is unavailable. Skipping.")
        else:
            detector = DetectabilityLoss(sampling_rate=args.sample_rate, frame_size=args.frame_size)
            print("Detectability metric enabled (libdetectability).")

    peaq_available = shutil.which("gstpeaq") is not None
    if args.peaq and not peaq_available:
        print("PEAQ requested but gstpeaq is not installed. PEAQ columns will be null.")

    rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for track_id in tracks:
            in_path = admm_dir / f"{track_id}_reference_clipped.npy"
            admm_path = admm_dir / f"{track_id}_admm_processed.npy"

            if not in_path.exists() or not admm_path.exists():
                print(f"Skipping {track_id} (missing reference/admm file).")
                continue

            reference = np.load(in_path).astype(np.float32).reshape(-1)
            admm = np.load(admm_path).astype(np.float32).reshape(-1)

            ref_t = to_tensor_1ct(reference, args.device)

            dsp_out_t, _ = dsp.process_batch(ref_t)
            dsp_out = tensor_to_numpy_1d(dsp_out_t)

            wave_out_t = waveunet(ref_t)
            wave_out = tensor_to_numpy_1d(wave_out_t)

            vae_out = vae(ref_t)
            vae_recon_t = vae_out["recon"] if isinstance(vae_out, dict) else vae_out
            vae_recon = tensor_to_numpy_1d(vae_recon_t)

            methods: List[Tuple[str, np.ndarray]] = [
                ("input_reference", reference),
                ("admm", admm),
                ("dsp_fast", dsp_out),
                ("waveunet_stage1", wave_out),
                ("vae_plain_cyclical", vae_recon),
            ]

            for method_name, output in methods:
                metric = compute_metrics(output, admm)
                metric["detectability_vs_admm"] = compute_detectability(
                    output,
                    admm,
                    detector,
                    frame_size=args.frame_size,
                )
                metric["peaq_odg_vs_admm"] = None
                rows.append(
                    {
                        "track_id": track_id,
                        "method": method_name,
                        **metric,
                    }
                )

    if not rows:
        raise RuntimeError("No rows computed. Check input files and track IDs.")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("outputs/admm_compare") / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    csv_path = out_dir / "results.csv"
    fieldnames = [
        "track_id",
        "method",
        "nmse_vs_admm",
        "mae_vs_admm",
        "output_peak",
        "output_rms_db",
        "detectability_vs_admm",
        "peaq_odg_vs_admm",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate by method for quick scanning.
    summary: Dict[str, Any] = {
        "detectability_enabled": bool(detector is not None),
        "detectability_available": bool(detectability_available),
        "peaq_requested": bool(args.peaq),
        "peaq_available": bool(peaq_available),
        "methods": {},
    }
    for method in sorted(set(r["method"] for r in rows)):
        subset = [r for r in rows if r["method"] == method]
        det_values = [r["detectability_vs_admm"] for r in subset if r.get("detectability_vs_admm") is not None]
        summary["methods"][method] = {
            "avg_nmse_vs_admm": float(np.mean([r["nmse_vs_admm"] for r in subset])),
            "avg_mae_vs_admm": float(np.mean([r["mae_vs_admm"] for r in subset])),
            "avg_output_peak": float(np.mean([r["output_peak"] for r in subset])),
            "avg_output_rms_db": float(np.mean([r["output_rms_db"] for r in subset])),
            "avg_detectability_vs_admm": float(np.mean(det_values)) if det_values else None,
            "num_samples": float(len(subset)),
        }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Tracks evaluated: {tracks}")
    print(f"Results CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print("Method averages:")
    for method, values in summary["methods"].items():
        det_txt = "None" if values["avg_detectability_vs_admm"] is None else f"{values['avg_detectability_vs_admm']:.6f}"
        print(
            f"  {method:18s} "
            f"nmse={values['avg_nmse_vs_admm']:.6f} "
            f"mae={values['avg_mae_vs_admm']:.6f} "
            f"peak={values['avg_output_peak']:.4f} "
            f"rms_db={values['avg_output_rms_db']:.2f} "
            f"det={det_txt}"
        )


if __name__ == "__main__":
    run()
