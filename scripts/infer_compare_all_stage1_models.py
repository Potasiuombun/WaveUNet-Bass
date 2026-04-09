#!/usr/bin/env python3
"""Run side-by-side inference for all trained Stage-1 models.

Outputs:
- original.wav
- input_normalized.wav
- <model_name>.wav for each model
- error_<model_name>.wav
- metrics.csv and metrics.json
- comparison_waveforms.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dsp.bandsplit import FixedBandSplitter, peak_safe
from src.dsp.ddsp_controls import apply_ddsp_controls
from src.dsp.fast_baseline import FastDSPBaseline
from src.models.band_controller import BandController
from src.models.ddsp_controller import DDSPController
from src.models.factory import create_model
from src.models.tiny_residual import TinyResidualModel, apply_residual_with_peak_safety
from src.models.vae_waveunet import VariationalWaveUNet


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_checkpoint_state(checkpoint_path: Path) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "model" in ckpt:
            return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")


def _load_audio_normalized(path: Path, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    y_orig = y.copy()
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / (peak + 1e-10)
    return y.astype(np.float32), y_orig.astype(np.float32)


def _frame_audio(x: np.ndarray, frame_size: int, hop_size: int) -> Tuple[np.ndarray, int]:
    if len(x) < frame_size:
        return np.zeros((0, frame_size), dtype=np.float32), len(x)
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop_size):
        frames.append(x[start : start + frame_size])
    return np.asarray(frames, dtype=np.float32), len(x)


def _ola(frames: np.ndarray, hop_size: int, frame_size: int, original_length: int) -> np.ndarray:
    if len(frames) == 0:
        return np.zeros((original_length,), dtype=np.float32)

    n_frames = frames.shape[0]
    out_len = (n_frames - 1) * hop_size + frame_size
    out_len = max(out_len, original_length)

    y = np.zeros((out_len,), dtype=np.float32)
    win = np.hanning(frame_size).astype(np.float32)
    wsum = np.zeros((out_len,), dtype=np.float32)

    for i in range(n_frames):
        s = i * hop_size
        y[s : s + frame_size] += frames[i] * win
        wsum[s : s + frame_size] += win

    wsum[wsum == 0.0] = 1.0
    y = y / wsum
    return y[:original_length]


def _run_wave_model(
    model: torch.nn.Module,
    frames: np.ndarray,
    device: str,
    batch_size: int,
    is_vae: bool = False,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = torch.from_numpy(frames[i : i + batch_size]).unsqueeze(1).to(device)
            if is_vae:
                # Deterministic decode for listening comparison: use z = mu.
                z, mu, _logvar, skips = model.encode(batch)
                _ = z
                out = model.decode(mu, skips, batch)
            else:
                out = model(batch)
                if isinstance(out, dict):
                    out = out["recon"]
            outputs.append(out.squeeze(1).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def _run_dsp_residual(
    model: TinyResidualModel,
    dsp: FastDSPBaseline,
    frames: np.ndarray,
    device: str,
    batch_size: int,
    peak_limit: float,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    model.eval()
    dsp.reset()
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = torch.from_numpy(frames[i : i + batch_size]).unsqueeze(1).to(device)
            dsp_out, _stats = dsp.process_batch(batch)
            residual = model(dsp_out)
            out = apply_residual_with_peak_safety(dsp_out, residual, peak_limit=peak_limit)
            outputs.append(out.squeeze(1).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def _run_band_controller(
    model: BandController,
    splitter: FixedBandSplitter,
    frames: np.ndarray,
    device: str,
    batch_size: int,
    peak_limit_value: float,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = torch.from_numpy(frames[i : i + batch_size]).unsqueeze(1).to(device)
            bands = splitter.analyze(batch)
            gains = model(batch)
            out = splitter.synthesize(splitter.apply_band_gains(bands, gains))
            out = peak_safe(out, peak_limit=peak_limit_value)
            outputs.append(out.squeeze(1).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def _run_ddsp_controller(
    model: DDSPController,
    splitter: FixedBandSplitter,
    frames: np.ndarray,
    device: str,
    batch_size: int,
    peak_limit_value: float,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = torch.from_numpy(frames[i : i + batch_size]).unsqueeze(1).to(device)
            controls = model(batch)
            out = apply_ddsp_controls(batch, splitter, controls, peak_limit=peak_limit_value)
            outputs.append(out.squeeze(1).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def _compute_metrics(y_hat: np.ndarray, y_ref: np.ndarray) -> Dict[str, float]:
    err = y_hat - y_ref
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err**2))
    nmse = float(mse / (np.mean(y_ref**2) + 1e-8))
    rmse = float(np.sqrt(mse))
    peak = float(np.max(np.abs(y_hat)))
    rms = float(np.sqrt(np.mean(y_hat**2)))
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "nmse": nmse,
        "peak": peak,
        "rms": rms,
    }


def _save_plot(output_dir: Path, sr: int, y_ref: np.ndarray, outputs: Dict[str, np.ndarray]) -> None:
    names = list(outputs.keys())
    n = len(names) + 1
    t = np.arange(len(y_ref)) / float(sr)

    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n), sharex=True)
    axes[0].plot(t, y_ref, linewidth=0.4)
    axes[0].set_title("Reference Input (Normalized)", fontweight="bold")
    axes[0].grid(alpha=0.25)

    for i, name in enumerate(names, start=1):
        axes[i].plot(t, outputs[name], linewidth=0.4)
        axes[i].set_title(name, fontweight="bold")
        axes[i].grid(alpha=0.25)

    axes[-1].set_xlabel("Time (s)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_waveforms.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer and compare all Stage-1 models")
    parser.add_argument(
        "--audio",
        default="/home/tudor/Documents/MasterThesis/fma_small_curated_20260408/fma_small/000/000890.mp3",
        help="Input audio path",
    )
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--frame-size", type=int, default=1024)
    parser.add_argument("--hop-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        default="inference_outputs/all_stage1_models",
        help="Output directory",
    )
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    root = Path(__file__).resolve().parent.parent
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = Path(args.audio)
    y_norm, y_orig = _load_audio_normalized(audio_path, sr=int(args.sr))
    frames, original_len = _frame_audio(y_norm, frame_size=int(args.frame_size), hop_size=int(args.hop_size))

    print("=" * 80)
    print("Stage-1 Model Comparison Inference")
    print("=" * 80)
    print(f"Audio: {audio_path}")
    print(f"Device: {device}")
    print(f"Samples: {len(y_norm)} | Duration: {len(y_norm)/args.sr:.2f}s | Frames: {len(frames)}")

    sf.write(output_dir / "original.wav", y_orig, int(args.sr))
    sf.write(output_dir / "input_normalized.wav", y_norm, int(args.sr))

    model_outputs: Dict[str, np.ndarray] = {}
    metrics_rows: List[Dict[str, Any]] = []

    # 1) WaveUNet baseline
    cfg_wave = _load_yaml(root / "configs/stage1_fma_curated_cli_norm.yaml")
    wave_model = create_model(cfg_wave["model"], device=device)
    wave_state = _load_checkpoint_state(root / "checkpoints/stage1_fma_curated_cli_norm/best.pth")
    wave_model.load_state_dict(wave_state)
    wave_frames = _run_wave_model(wave_model, frames, device=device, batch_size=int(args.batch_size), is_vae=False)
    y_wave = _ola(wave_frames, int(args.hop_size), int(args.frame_size), original_len)
    model_outputs["waveunet"] = y_wave

    # 2) VAE WaveUNet (deterministic decode using mu)
    cfg_vae = _load_yaml(root / "configs/stage1_vae_fma_curated_cli_norm.yaml")
    vae_model = create_model(cfg_vae["model"], device=device)
    vae_state = _load_checkpoint_state(root / "checkpoints/stage1_vae_fma_curated_cli_norm/best_stage3.pth")
    vae_model.load_state_dict(vae_state)
    if not isinstance(vae_model, VariationalWaveUNet):
        raise RuntimeError("Expected VariationalWaveUNet for VAE config")
    vae_frames = _run_wave_model(vae_model, frames, device=device, batch_size=int(args.batch_size), is_vae=True)
    y_vae = _ola(vae_frames, int(args.hop_size), int(args.frame_size), original_len)
    model_outputs["vae_waveunet"] = y_vae

    # 3) DSP + Tiny Residual
    cfg_dsp = _load_yaml(root / "configs/stage1_dsp_residual_fma_curated_cli_norm.yaml")
    residual_model = TinyResidualModel(
        hidden_channels=int(cfg_dsp["model"]["hidden_channels"]),
        num_layers=int(cfg_dsp["model"]["num_layers"]),
        kernel_size=int(cfg_dsp["model"]["kernel_size"]),
        activation=str(cfg_dsp["model"]["activation"]),
        max_residual=float(cfg_dsp["model"]["max_residual"]),
    ).to(device)
    latest_dsp = _load_yaml(root / "outputs/stage1_dsp_residual/LATEST_DSP_RESIDUAL.json")
    residual_state = _load_checkpoint_state(root / latest_dsp["checkpoint_best"])
    residual_model.load_state_dict(residual_state)
    dsp_processor = FastDSPBaseline(cfg_dsp["dsp"], device=device)
    dsp_frames = _run_dsp_residual(
        residual_model,
        dsp_processor,
        frames,
        device=device,
        batch_size=int(args.batch_size),
        peak_limit=float(cfg_dsp["dsp"].get("peak_limit", 0.99)),
    )
    y_dsp = _ola(dsp_frames, int(args.hop_size), int(args.frame_size), original_len)
    model_outputs["dsp_tiny_residual"] = y_dsp

    # 4) Band controller
    cfg_band = _load_yaml(root / "configs/stage1_band_controller_fma_curated_cli_norm.yaml")
    band_model = BandController(
        num_bands=len(cfg_band["dsp"]["bands_hz"]),
        hidden_channels=int(cfg_band["model"]["hidden_channels"]),
        max_gain_db=float(cfg_band["model"]["max_gain_db"]),
        min_gain_db=float(cfg_band["model"]["min_gain_db"]),
    ).to(device)
    latest_band = _load_yaml(root / "outputs/stage1_band_controller/LATEST_BAND_CONTROLLER.json")
    band_state = _load_checkpoint_state(root / latest_band["checkpoint_best"])
    band_model.load_state_dict(band_state)
    band_splitter = FixedBandSplitter(sample_rate=int(cfg_band["data"]["sample_rate"]), bands_hz=cfg_band["dsp"]["bands_hz"])
    band_frames = _run_band_controller(
        band_model,
        band_splitter,
        frames,
        device=device,
        batch_size=int(args.batch_size),
        peak_limit_value=float(cfg_band["dsp"].get("peak_limit", 0.99)),
    )
    y_band = _ola(band_frames, int(args.hop_size), int(args.frame_size), original_len)
    model_outputs["band_controller"] = y_band

    # 5) DDSP controller
    cfg_ddsp = _load_yaml(root / "configs/stage1_ddsp_controller_fma_curated_cli_norm.yaml")
    ddsp_model = DDSPController(
        num_bands=len(cfg_ddsp["dsp"]["bands_hz"]),
        hidden_channels=int(cfg_ddsp["model"]["hidden_channels"]),
        max_gain_db=float(cfg_ddsp["model"]["max_gain_db"]),
        min_gain_db=float(cfg_ddsp["model"]["min_gain_db"]),
        max_tilt_db=float(cfg_ddsp["model"]["max_tilt_db"]),
        envelope_enabled=bool(cfg_ddsp["model"]["envelope_enabled"]),
        min_envelope=float(cfg_ddsp["model"]["min_envelope"]),
        max_envelope=float(cfg_ddsp["model"]["max_envelope"]),
    ).to(device)
    latest_ddsp = _load_yaml(root / "outputs/stage1_ddsp_controller/LATEST_DDSP_CONTROLLER.json")
    ddsp_state = _load_checkpoint_state(root / latest_ddsp["checkpoint_best"])
    ddsp_model.load_state_dict(ddsp_state)
    ddsp_splitter = FixedBandSplitter(sample_rate=int(cfg_ddsp["data"]["sample_rate"]), bands_hz=cfg_ddsp["dsp"]["bands_hz"])
    ddsp_frames = _run_ddsp_controller(
        ddsp_model,
        ddsp_splitter,
        frames,
        device=device,
        batch_size=int(args.batch_size),
        peak_limit_value=float(cfg_ddsp["dsp"].get("peak_limit", 0.99)),
    )
    y_ddsp = _ola(ddsp_frames, int(args.hop_size), int(args.frame_size), original_len)
    model_outputs["ddsp_controller"] = y_ddsp

    for name, y_hat in model_outputs.items():
        sf.write(output_dir / f"{name}.wav", y_hat, int(args.sr))
        sf.write(output_dir / f"error_{name}.wav", y_hat - y_norm, int(args.sr))

        row = {"model": name}
        row.update(_compute_metrics(y_hat, y_norm))
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("nmse", ascending=True)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_rows, f, indent=2)

    _save_plot(output_dir, int(args.sr), y_norm, model_outputs)

    print("\nSaved outputs:")
    print(f"  {output_dir / 'input_normalized.wav'}")
    print(f"  {output_dir / 'original.wav'}")
    for name in model_outputs:
        print(f"  {output_dir / (name + '.wav')}")
    print(f"  {output_dir / 'metrics.csv'}")
    print(f"  {output_dir / 'comparison_waveforms.png'}")

    print("\nRanking by NMSE (lower is better):")
    for i, row in enumerate(metrics_df.to_dict(orient="records"), start=1):
        print(f"  {i}. {row['model']}: nmse={row['nmse']:.6f}, mae={row['mae']:.6f}")


if __name__ == "__main__":
    main()
