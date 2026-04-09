#!/usr/bin/env python3
"""Stage-2 listening inference with DSP synthesis and metrics.

Generates:
- per-model output WAV and error WAV
- NMSE, MAE, detectability
- PEAQ ODG (if PQevalAudio is available)
- comparison plots
"""

from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.io import wavfile

warnings.filterwarnings("ignore")

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dsp.bandsplit import FixedBandSplitter, peak_safe
from src.dsp.ddsp_controls import apply_ddsp_controls
from src.dsp.fast_baseline import FastDSPBaseline
from src.losses.perceptual import DetectabilityLossWrapper
from src.models.band_controller import BandController
from src.models.ddsp_controller import DDSPController
from src.models.factory import create_model
from src.models.tiny_residual import TinyResidualModel, apply_residual_with_peak_safety
from src.training.metrics import mae, nmse


def _frame_audio(x: np.ndarray, frame_size: int = 1024, hop_size: int = 512) -> Tuple[np.ndarray, int]:
    if len(x) < frame_size:
        return np.zeros((0, frame_size), dtype=np.float32), len(x)
    frames = [x[s : s + frame_size] for s in range(0, len(x) - frame_size + 1, hop_size)]
    return np.asarray(frames, dtype=np.float32), len(x)


def _ola(frames: np.ndarray, hop_size: int = 512, frame_size: int = 1024, orig_len: int = 0) -> np.ndarray:
    if len(frames) == 0:
        return np.zeros((orig_len,), dtype=np.float32)

    n_frames = frames.shape[0]
    out_len = max((n_frames - 1) * hop_size + frame_size, orig_len)

    y = np.zeros((out_len,), dtype=np.float32)
    win = np.hanning(frame_size).astype(np.float32)
    wsum = np.zeros((out_len,), dtype=np.float32)

    for i in range(n_frames):
        s = i * hop_size
        y[s : s + frame_size] += frames[i] * win
        wsum[s : s + frame_size] += win

    wsum[wsum == 0.0] = 1.0
    y = y / wsum
    return y[:orig_len]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_checkpoint(path: Path, device: str) -> Dict[str, Any]:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        if "model_state_dict" in state:
            return state["model_state_dict"]
        if "model" in state:
            return state["model"]
        if "state_dict" in state:
            return state["state_dict"]
    if isinstance(state, dict):
        return state
    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def _cfg_sample_rate(cfg: Dict[str, Any], default_sr: int = 44100) -> int:
    data = cfg.get("data", {})
    if "sample_rate" in data:
        return int(data["sample_rate"])
    if "sr" in data:
        return int(data["sr"])
    return int(default_sr)


def _load_stage2_bundles(root: Path, device: str) -> Tuple[Dict[str, Dict[str, Any]], int]:
    bundles: Dict[str, Dict[str, Any]] = {}

    # WaveUNet
    cfg = _load_yaml(root / "configs/stage1_fma_curated_cli_norm.yaml")
    model = create_model(cfg["model"], device=device).eval()
    model.load_state_dict(_load_checkpoint(root / "checkpoints/stage2_detectability_waveunet/best_stage2.pth", device))
    bundles["waveunet"] = {
        "type": "waveunet",
        "model": model,
        "sample_rate": _cfg_sample_rate(cfg),
    }

    # VAE WaveUNet
    cfg = _load_yaml(root / "configs/stage1_vae_fma_curated_cli_norm.yaml")
    model = create_model(cfg["model"], device=device).eval()
    model.load_state_dict(
        _load_checkpoint(root / "checkpoints/stage2_detectability_vae_waveunet/best_stage2.pth", device)
    )
    bundles["vae_waveunet"] = {
        "type": "vae_waveunet",
        "model": model,
        "sample_rate": _cfg_sample_rate(cfg),
    }

    # DSP residual
    cfg = _load_yaml(root / "configs/stage1_dsp_residual_fma_curated_cli_norm.yaml")
    model = TinyResidualModel(
        hidden_channels=int(cfg["model"]["hidden_channels"]),
        num_layers=int(cfg["model"]["num_layers"]),
        kernel_size=int(cfg["model"]["kernel_size"]),
        activation=str(cfg["model"]["activation"]),
        max_residual=float(cfg["model"]["max_residual"]),
    ).to(device).eval()
    model.load_state_dict(
        _load_checkpoint(root / "checkpoints/stage2_detectability_dsp_tiny_residual/best_stage2.pth", device)
    )
    bundles["dsp_tiny_residual"] = {
        "type": "dsp_tiny_residual",
        "model": model,
        "dsp": FastDSPBaseline(cfg["dsp"], device=device),
        "peak_limit": float(cfg["dsp"].get("peak_limit", 0.99)),
        "sample_rate": _cfg_sample_rate(cfg),
    }

    # Band controller
    cfg = _load_yaml(root / "configs/stage1_band_controller_fma_curated_cli_norm.yaml")
    model = BandController(
        num_bands=len(cfg["dsp"]["bands_hz"]),
        hidden_channels=int(cfg["model"]["hidden_channels"]),
        max_gain_db=float(cfg["model"]["max_gain_db"]),
        min_gain_db=float(cfg["model"]["min_gain_db"]),
    ).to(device).eval()
    model.load_state_dict(
        _load_checkpoint(root / "checkpoints/stage2_detectability_band_controller/best_stage2.pth", device)
    )
    bundles["band_controller"] = {
        "type": "band_controller",
        "model": model,
        "splitter": FixedBandSplitter(
            sample_rate=_cfg_sample_rate(cfg), bands_hz=cfg["dsp"]["bands_hz"]
        ),
        "peak_limit": float(cfg["dsp"].get("peak_limit", 0.99)),
        "sample_rate": _cfg_sample_rate(cfg),
    }

    # DDSP controller
    cfg = _load_yaml(root / "configs/stage1_ddsp_controller_fma_curated_cli_norm.yaml")
    model = DDSPController(
        num_bands=len(cfg["dsp"]["bands_hz"]),
        hidden_channels=int(cfg["model"]["hidden_channels"]),
        max_gain_db=float(cfg["model"]["max_gain_db"]),
        min_gain_db=float(cfg["model"]["min_gain_db"]),
        max_tilt_db=float(cfg["model"]["max_tilt_db"]),
        envelope_enabled=bool(cfg["model"]["envelope_enabled"]),
        min_envelope=float(cfg["model"]["min_envelope"]),
        max_envelope=float(cfg["model"]["max_envelope"]),
    ).to(device).eval()
    model.load_state_dict(
        _load_checkpoint(root / "checkpoints/stage2_detectability_ddsp_controller/best_stage2.pth", device)
    )
    bundles["ddsp_controller"] = {
        "type": "ddsp_controller",
        "model": model,
        "splitter": FixedBandSplitter(
            sample_rate=_cfg_sample_rate(cfg), bands_hz=cfg["dsp"]["bands_hz"]
        ),
        "peak_limit": float(cfg["dsp"].get("peak_limit", 0.99)),
        "sample_rate": _cfg_sample_rate(cfg),
    }

    sample_rate = int(next(iter(bundles.values()))["sample_rate"])
    return bundles, sample_rate


def _model_forward(bundle: Dict[str, Any], x: torch.Tensor) -> torch.Tensor:
    model_type = bundle["type"]
    model = bundle["model"]

    if model_type == "waveunet":
        return model(x)

    if model_type == "vae_waveunet":
        out = model(x)
        if isinstance(out, dict):
            return out.get("recon", out.get("reconstruction"))
        return out

    if model_type == "dsp_tiny_residual":
        dsp: FastDSPBaseline = bundle["dsp"]
        dsp_out, _ = dsp.process_batch(x)
        residual = model(dsp_out)
        return apply_residual_with_peak_safety(dsp_out, residual, peak_limit=float(bundle["peak_limit"]))

    if model_type == "band_controller":
        splitter: FixedBandSplitter = bundle["splitter"]
        bands = splitter.analyze(x)
        gains = model(x)  # linear gains [N,BANDS]
        y = splitter.synthesize(splitter.apply_band_gains(bands, gains))
        return peak_safe(y, peak_limit=float(bundle["peak_limit"]))

    if model_type == "ddsp_controller":
        splitter: FixedBandSplitter = bundle["splitter"]
        controls = model(x)  # dict with band_gain_db, spectral_tilt_db, gain_envelope(optional)
        return apply_ddsp_controls(x, splitter, controls, peak_limit=float(bundle["peak_limit"]))

    raise ValueError(f"Unhandled model type: {model_type}")


def compute_detectability(y_ref: np.ndarray, y_pred: np.ndarray, sr: int) -> float:
    # Primary: libdetectability wrapper used in training.
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loss_fn = DetectabilityLossWrapper(enabled=True, sample_rate=sr, frame_size=1024).to(device)
        y_ref_t = torch.from_numpy(y_ref).float().unsqueeze(0).to(device)
        y_pred_t = torch.from_numpy(y_pred).float().unsqueeze(0).to(device)
        with torch.no_grad():
            d = loss_fn(y_pred_t, y_ref_t)
        return float(d.item() if isinstance(d, torch.Tensor) else d)
    except Exception:
        pass

    # Fallback: controlmethods perceptual detectability over overlapping blocks.
    try:
        cm_root = Path(__file__).resolve().parents[2] / "controlmethods-python"
        if str(cm_root) not in sys.path:
            sys.path.insert(0, str(cm_root))
        from controlmethods.perceptual_model import PerceptualModel  # type: ignore

        block_size = 1024
        hop = 512
        pm = PerceptualModel(block_size=block_size, fs=sr)

        values = []
        max_start = len(y_ref) - block_size
        for s in range(0, max_start + 1, hop):
            ref_block = y_ref[s : s + block_size]
            pred_block = y_pred[s : s + block_size]
            err_block = pred_block - ref_block
            pm.determine_squared_weighting_curve(ref_block.astype(np.float64))
            d = pm.evaluate_detectability(err_block.astype(np.float64))
            values.append(float(d))

        if values:
            return float(np.mean(values))
        return -1.0
    except Exception:
        return -1.0


_PEAQ_BIN = Path(__file__).resolve().parents[2] / "peaq-python-simple-main/AFsp-v9r0/bin/PQevalAudio"
_PEAQ_SR = 48000  # PQevalAudio requires 48 kHz


def compute_peaq_odg(y_ref: np.ndarray, y_pred: np.ndarray, sr: int) -> float:
    """Compute PEAQ ODG via the bundled PQevalAudio binary (resamples to 48 kHz)."""
    import re
    import subprocess

    if not _PEAQ_BIN.exists():
        return float("nan")

    try:
        # Resample to required 48 kHz
        if sr != _PEAQ_SR:
            y_ref_48 = librosa.resample(y_ref, orig_sr=sr, target_sr=_PEAQ_SR)
            y_pred_48 = librosa.resample(y_pred, orig_sr=sr, target_sr=_PEAQ_SR)
        else:
            y_ref_48, y_pred_48 = y_ref, y_pred

        with tempfile.NamedTemporaryFile(suffix="_ref.wav", delete=False) as f_ref, \
             tempfile.NamedTemporaryFile(suffix="_test.wav", delete=False) as f_test:
            ref_path = Path(f_ref.name)
            test_path = Path(f_test.name)

        wavfile.write(str(ref_path), _PEAQ_SR, (np.clip(y_ref_48, -1, 1) * 32767).astype(np.int16))
        wavfile.write(str(test_path), _PEAQ_SR, (np.clip(y_pred_48, -1, 1) * 32767).astype(np.int16))

        result = subprocess.run(
            [str(_PEAQ_BIN), "--levelSPL", "85", str(ref_path), str(test_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )

        odg = float("nan")
        for line in result.stdout.split("\n"):
            if "Objective Difference Grade" in line:
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if nums:
                    odg = float(nums[0])
                break

        return odg
    except Exception:
        return float("nan")
    finally:
        try:
            ref_path.unlink(missing_ok=True)
            test_path.unlink(missing_ok=True)
        except Exception:
            pass


def process_audio(audio_path: Path, device: str, outdir: Path, bundles: Dict[str, Dict[str, Any]], sr: int) -> Dict[str, Any]:
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    y = y.astype(np.float32)

    frames, orig_len = _frame_audio(y, frame_size=1024, hop_size=512)

    inp_wav = outdir / "input_normalized.wav"
    wavfile.write(str(inp_wav), sr, (np.clip(y, -1, 1) * 32767).astype(np.int16))

    results: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "audio": str(audio_path),
        "sample_rate": sr,
        "duration_sec": len(y) / sr,
        "input_wav": str(inp_wav),
        "models": {},
    }

    for model_name, bundle in bundles.items():
        print(f"  {model_name}...", end=" ", flush=True)
        try:
            with torch.no_grad():
                frame_outs = []
                for frame in frames:
                    x = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0).to(device)
                    y_hat = _model_forward(bundle, x)
                    frame_outs.append(y_hat.squeeze(0).squeeze(0).detach().cpu().numpy())

            y_out = _ola(np.asarray(frame_outs, dtype=np.float32), hop_size=512, frame_size=1024, orig_len=orig_len)
            peak_out = np.max(np.abs(y_out))
            if peak_out > 0:
                y_out = y_out / peak_out

            nmse_val = float(nmse(torch.from_numpy(y).float(), torch.from_numpy(y_out).float()))
            mae_val = float(mae(torch.from_numpy(y).float(), torch.from_numpy(y_out).float()))
            detect_val = compute_detectability(y, y_out, sr)
            peaq_odg = compute_peaq_odg(y, y_out, sr)

            out_wav = outdir / f"{model_name}_output.wav"
            err_wav = outdir / f"{model_name}_error.wav"
            wavfile.write(str(out_wav), sr, (np.clip(y_out, -1, 1) * 32767).astype(np.int16))
            wavfile.write(str(err_wav), sr, (np.clip(y - y_out, -1, 1) * 32767).astype(np.int16))

            results["models"][model_name] = {
                "output_wav": str(out_wav),
                "error_wav": str(err_wav),
                "nmse": nmse_val,
                "mae": mae_val,
                "detectability": detect_val,
                "peaq_odg": peaq_odg,
            }

            print(f"NMSE={nmse_val:.6f} Detect={detect_val:.6f} PEAQ_ODG={peaq_odg:.3f}")
        except Exception as e:
            print(f"Error: {e}")
            results["models"][model_name] = {"error": str(e)}

    return results


def generate_plots(results: Dict[str, Any], outdir: Path) -> None:
    _, y_inp = wavfile.read(results["input_wav"])
    y_inp = y_inp.astype(np.float32) / 32767.0

    names = [k for k in results["models"] if "error" not in results["models"][k]]
    if not names:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    nmse_v = [results["models"][m].get("nmse", 0.0) for m in names]
    det_v = [results["models"][m].get("detectability", 0.0) for m in names]
    peaq_v = [results["models"][m].get("peaq_odg", np.nan) for m in names]

    axes[0].bar(names, nmse_v, color="steelblue")
    axes[0].set_ylabel("NMSE")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(names, det_v, color="green")
    axes[1].set_ylabel("Detectability")
    axes[1].tick_params(axis="x", rotation=45)

    axes[2].bar(names, peaq_v, color="coral")
    axes[2].set_ylabel("PEAQ ODG")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(outdir / "metrics_comparison.png", dpi=150)
    plt.close()
    print("✓ Metrics plot")

    n = len(names)
    fig, axes = plt.subplots(n + 1, 1, figsize=(14, n * 2))
    if n == 1:
        axes = [axes]
    t = np.arange(len(y_inp)) / results["sample_rate"]

    axes[0].plot(t, y_inp, linewidth=0.5, color="black")
    axes[0].set_ylabel("Input")

    for i, name in enumerate(names):
        _, y = wavfile.read(results["models"][name]["output_wav"])
        y = y.astype(np.float32) / 32767.0
        axes[i + 1].plot(t, y, linewidth=0.5, color="steelblue")
        d = results["models"][name].get("detectability", 0.0)
        axes[i + 1].set_ylabel(name)
        axes[i + 1].set_title(f"{name} (Detect={d:.4f})")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(outdir / "waveform_comparison.png", dpi=150)
    plt.close()
    print("✓ Waveform plot")

    fig, axes = plt.subplots(n + 1, 1, figsize=(14, n * 2))
    if n == 1:
        axes = [axes]

    s = librosa.magphase(librosa.stft(y_inp, n_fft=2048, hop_length=512))[0]
    s_db = librosa.power_to_db(s, ref=np.max)
    im = axes[0].imshow(s_db, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_ylabel("Freq")
    axes[0].set_title("Input")
    plt.colorbar(im, ax=axes[0])

    for i, name in enumerate(names):
        _, y = wavfile.read(results["models"][name]["output_wav"])
        y = y.astype(np.float32) / 32767.0
        s = librosa.magphase(librosa.stft(y, n_fft=2048, hop_length=512))[0]
        s_db = librosa.power_to_db(s, ref=np.max)
        im = axes[i + 1].imshow(s_db, aspect="auto", origin="lower", cmap="viridis")
        axes[i + 1].set_ylabel("Freq")
        axes[i + 1].set_title(name)
        plt.colorbar(im, ax=axes[i + 1])

    axes[-1].set_xlabel("Time (frame)")
    plt.tight_layout()
    plt.savefig(outdir / "spectrogram_comparison.png", dpi=150)
    plt.close()
    print("✓ Spectrogram plot")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audio", type=str, required=True, help="Absolute or relative path to input WAV")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default="outputs/stage2_listening")
    args = p.parse_args()

    device = "cuda" if (args.device.lower() == "cuda" and torch.cuda.is_available()) else "cpu"
    root = Path(__file__).parent.parent
    outdir = root / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    audio_path = Path(args.audio)
    if not audio_path.is_absolute():
        audio_path = root / audio_path
    audio_path = audio_path.resolve()

    if not audio_path.exists():
        print(f"✗ Audio not found: {audio_path}")
        return

    print("Loading Stage 2 model bundles with DSP helpers...")
    bundles, sr = _load_stage2_bundles(root, device)
    print(f"Loaded {len(bundles)} models | sample_rate={sr}")

    print(f"\nProcessing: {audio_path}\n")
    results = process_audio(audio_path, device, outdir, bundles, sr)

    res_json = outdir / "results.json"
    with open(res_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results JSON: {res_json}")

    generate_plots(results, outdir)

    print("\n" + "=" * 80)
    print("STAGE-2 SUMMARY (sorted by detectability)")
    print("=" * 80)
    for name in sorted(bundles.keys(), key=lambda x: results["models"].get(x, {}).get("detectability", 999.0)):
        m = results["models"].get(name, {})
        if "error" not in m:
            print(
                f"{name:20s}: NMSE={m['nmse']:.6f}  Detect={m['detectability']:.6f}  PEAQ_ODG={m['peaq_odg']:.3f}"
            )
        else:
            print(f"{name:20s}: ERROR={m['error']}")


if __name__ == "__main__":
    main()
