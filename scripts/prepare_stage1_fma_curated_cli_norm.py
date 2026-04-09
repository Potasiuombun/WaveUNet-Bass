#!/usr/bin/env python3
"""Prepare Stage-1 identity dataset from curated FMA-small with CLI-like normalization.

Normalization order follows the CLI tool:
1) load track waveform
2) convert to mono
3) peak-normalize whole track to [-1, 1]
4) frame waveform
5) use normalized frames as both input and target for Stage-1 reconstruction

This script uses curated split manifests to avoid leakage.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import pandas as pd
import torch


def _read_ids(path: Path) -> List[str]:
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        ids.append(s)
    return ids


def _load_mp3_mono(path: Path, target_sr: int) -> torch.Tensor:
    y, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return torch.from_numpy(y).float()


def _frame_audio(x: torch.Tensor, frame_size: int, hop_size: int) -> torch.Tensor:
    if x.ndim != 1:
        raise ValueError(f"Expected 1D waveform, got shape {tuple(x.shape)}")
    T = int(x.shape[0])
    if T < frame_size:
        return torch.empty((0, frame_size), dtype=x.dtype)
    frames = []
    for start in range(0, T - frame_size + 1, hop_size):
        frames.append(x[start : start + frame_size])
    return torch.stack(frames, dim=0)


def _track_path(root: Path, track_id: str) -> Path:
    tid = track_id.zfill(6)
    return root / tid[:3] / f"{tid}.mp3"


def _sample_frames(frames: torch.Tensor, max_frames_per_track: int, rng: np.random.Generator) -> torch.Tensor:
    n = int(frames.shape[0])
    if max_frames_per_track <= 0 or n <= max_frames_per_track:
        return frames
    idx = rng.choice(n, size=max_frames_per_track, replace=False)
    idx = np.sort(idx)
    return frames[torch.from_numpy(idx)]


def _build_rows(
    split_name: str,
    track_ids: List[str],
    audio_root: Path,
    target_sr: int,
    frame_size: int,
    hop_size: int,
    max_frames_per_track: int,
    seed: int,
) -> List[Dict]:
    rows: List[Dict] = []
    rng = np.random.default_rng(seed)

    for track_id in track_ids:
        p = _track_path(audio_root, track_id)
        if not p.exists():
            continue

        wav = _load_mp3_mono(p, target_sr=target_sr).float()

        # CLI normalization happens before optimization.
        peak = torch.max(torch.abs(wav))
        if float(peak) > 0.0:
            wav = wav / (peak + 1e-10)

        frames = _frame_audio(wav, frame_size=frame_size, hop_size=hop_size)
        if frames.shape[0] == 0:
            continue

        frames = _sample_frames(frames, max_frames_per_track=max_frames_per_track, rng=rng)

        for i in range(int(frames.shape[0])):
            fr = frames[i].cpu().numpy().astype(np.float32)
            rows.append(
                {
                    "input": fr,
                    "target": fr.copy(),
                    "track_id": track_id,
                    "frame_index": int(i),
                    "source_input_path": str(p),
                    "source_target_path": str(p),
                    "split": split_name,
                    "normalization_mode": "cli_pre_peak",
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stage-1 curated FMA dataset with CLI-like normalization")
    parser.add_argument(
        "--audio-root",
        default="/home/tudor/Documents/MasterThesis/fma_small_curated_20260408/fma_small",
        help="Root containing curated FMA mp3 files",
    )
    parser.add_argument(
        "--manifest-root",
        default="/home/tudor/Documents/MasterThesis/fma_small_curated_20260408/manifests",
        help="Root containing curated split id lists",
    )
    parser.add_argument(
        "--output",
        default="/home/tudor/Documents/MasterThesis/WaveUNet-Bass/datasets/stage1_fma_curated_cli_norm.pkl",
        help="Output serialized dataset path (.pkl)",
    )
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--frame-size", type=int, default=1024)
    parser.add_argument("--hop-size", type=int, default=512)
    parser.add_argument(
        "--max-frames-per-track",
        type=int,
        default=16,
        help="Cap frames per track for practical dataset size (0 disables cap)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    audio_root = Path(args.audio_root)
    manifest_root = Path(args.manifest_root)
    output_path = Path(args.output)

    train_ids = _read_ids(manifest_root / "training_track_ids.txt")
    val_ids = _read_ids(manifest_root / "validation_track_ids.txt")
    test_ids = _read_ids(manifest_root / "test_track_ids.txt")

    rows = []
    rows.extend(
        _build_rows(
            split_name="train",
            track_ids=train_ids,
            audio_root=audio_root,
            target_sr=int(args.sr),
            frame_size=int(args.frame_size),
            hop_size=int(args.hop_size),
            max_frames_per_track=int(args.max_frames_per_track),
            seed=int(args.seed),
        )
    )
    rows.extend(
        _build_rows(
            split_name="val",
            track_ids=val_ids,
            audio_root=audio_root,
            target_sr=int(args.sr),
            frame_size=int(args.frame_size),
            hop_size=int(args.hop_size),
            max_frames_per_track=int(args.max_frames_per_track),
            seed=int(args.seed) + 1,
        )
    )
    rows.extend(
        _build_rows(
            split_name="test",
            track_ids=test_ids,
            audio_root=audio_root,
            target_sr=int(args.sr),
            frame_size=int(args.frame_size),
            hop_size=int(args.hop_size),
            max_frames_per_track=int(args.max_frames_per_track),
            seed=int(args.seed) + 2,
        )
    )

    if not rows:
        raise RuntimeError("No rows produced. Check audio root and manifests.")

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(df, output_path)

    split_counts = df["split"].value_counts().to_dict()
    tracks = df.groupby("split")["track_id"].nunique().to_dict()

    print(f"Wrote dataset: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Split frame counts: {split_counts}")
    print(f"Unique tracks by split: {tracks}")
    print(f"Normalization mode: cli_pre_peak (per-track, pre-framing)")


if __name__ == "__main__":
    main()
