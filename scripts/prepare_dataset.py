#!/usr/bin/env python3
"""Prepare and serialize datasets to pickle/parquet format.

Supports two source types:

  raw_files  (default)
      Scans a flat directory for paired WAV/NPY files matched by glob patterns.
      Use ``--input-pattern`` / ``--target-pattern`` to configure.

  temp_data_outputs
      Scans ``*_output`` sub-folders under ``--data-root``, pairs input/target
      ``.npy`` files by a leading digit track-ID prefix, and attaches optional
      auxiliary files (wav, rescaled npy).  Use ``--input-suffix`` /
      ``--target-suffix`` to configure.

      Auto-selected when ``--input-suffix`` or ``--target-suffix`` is given.
"""
import argparse
import sys
from pathlib import Path
import pickle
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.serialized import build_dataset_from_raw, build_dataset_from_temp_data_outputs


def _warn_suspicious_metadata_uniqueness(df: pd.DataFrame, context: str = "") -> None:
    """Warn when per-track metadata path columns look suspiciously sparse.

    For rich datasets, expected path columns should usually vary roughly with
    track count. This does not fail the build; it surfaces potential bugs or
    missing source files.
    """
    if "track_id" not in df.columns:
        return

    track_count = int(df["track_id"].nunique())
    if track_count <= 1:
        return

    path_cols = [
        "source_input_path",
        "source_target_path",
        "reference_wav_path",
        "processed_wav_path",
        "rescaled_npy_path",
    ]

    label = f" ({context})" if context else ""
    for col in path_cols:
        if col not in df.columns:
            continue

        non_null = df[col].notna()
        tracks_with_value = int(df.loc[non_null, "track_id"].nunique())
        ratio = tracks_with_value / track_count

        # Suspicious: fewer than half the tracks have a value for this path.
        if tracks_with_value > 0 and ratio < 0.5:
            print(
                f"  ⚠ Metadata warning{label}: '{col}' present for "
                f"{tracks_with_value}/{track_count} tracks ({ratio:.1%})."
            )
        elif tracks_with_value == 0:
            print(
                f"  ⚠ Metadata warning{label}: '{col}' has no non-null values."
            )


def main():
    parser = argparse.ArgumentParser(
        description="Build and serialize dataset from raw audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── required args ──────────────────────────────────────────────────────
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root directory to scan for audio files"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output file (.pkl or .parquet)"
    )

    # ── source type ────────────────────────────────────────────────────────
    parser.add_argument(
        "--source-type",
        choices=["raw_files", "temp_data_outputs"],
        default=None,
        help=(
            "Dataset source type.  Auto-detected from other flags when omitted: "
            "temp_data_outputs is selected when --input-suffix or --target-suffix "
            "is provided; otherwise raw_files is used."
        )
    )

    # ── temp_data_outputs ─────────────────────────────────────────────────
    parser.add_argument(
        "--input-suffix",
        default=None,
        help=(
            "[temp_data_outputs] Filename suffix for input NPY files "
            "(default: _reference_clipped.npy)"
        )
    )
    parser.add_argument(
        "--target-suffix",
        default=None,
        help=(
            "[temp_data_outputs] Filename suffix for target NPY files "
            "(default: _admm_processed.npy)"
        )
    )
    parser.add_argument(
        "--split-by",
        choices=["track_id", "group_id"],
        default="track_id",
        help="[temp_data_outputs] Split grouping key (default: track_id)"
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="[temp_data_outputs] Limit number of *_output groups included"
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="[temp_data_outputs] Limit total complete tracks included"
    )
    parser.add_argument(
        "--max-frames-per-track",
        type=int,
        default=None,
        help="[temp_data_outputs] Cap kept frames per track after threshold filtering"
    )

    # ── raw_files ────────────────────────────────────────────────────────
    parser.add_argument(
        "--input-pattern",
        default="*reference*",
        help="[raw_files] Glob pattern for input files (default: *reference*)"
    )
    parser.add_argument(
        "--target-pattern",
        default="*processed*",
        help="[raw_files] Glob pattern for target files (default: *processed*)"
    )

    # ── shared framing / normalization ────────────────────────────────────
    parser.add_argument(
        "--frame-size",
        type=int,
        default=1024,
        help="Frame size in samples (default: 1024)"
    )
    parser.add_argument(
        "--hop-size",
        type=int,
        default=512,
        help="Hop size in samples (default: 512)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Amplitude threshold: keep frame if max(|input|) >= threshold (default: 0.0)"
    )
    parser.add_argument(
        "--normalization",
        choices=["none", "peak_per_track", "rms_per_track"],
        default="none",
        help="Per-track normalization mode (default: none)"
    )

    # ── split ratios / seed ───────────────────────────────────────────────
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Val split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for track splitting (default: 42)"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=48000,
        help="[raw_files] Expected sample rate (default: 48000)"
    )

    args = parser.parse_args()

    # ── resolve source type ───────────────────────────────────────────────
    if args.source_type is None:
        if args.input_suffix is not None or args.target_suffix is not None:
            source_type = "temp_data_outputs"
            print("ℹ️  Auto-selected --source-type temp_data_outputs "
                  "(--input-suffix / --target-suffix detected)")
        else:
            source_type = "raw_files"
    else:
        source_type = args.source_type

    # ── validate common args ──────────────────────────────────────────────
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"❌ Error: --data-root does not exist: {data_root}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"❌ Error: split ratios must sum to 1.0, got {total_ratio:.4f}")
        sys.exit(1)

    split_tuple = (args.train_ratio, args.val_ratio, args.test_ratio)

    # ── dispatch to builder ───────────────────────────────────────────────
    try:
        if source_type == "temp_data_outputs":
            input_suffix  = args.input_suffix  or "_reference_clipped.npy"
            target_suffix = args.target_suffix or "_admm_processed.npy"

            print(f"\nBuilding dataset from {data_root} (source_type=temp_data_outputs)")
            print(f"   Input suffix  : {input_suffix}")
            print(f"   Target suffix : {target_suffix}")
            print(f"   Frame size    : {args.frame_size}  |  Hop size: {args.hop_size}")
            print(f"   Threshold     : {args.threshold}")
            print(f"   Normalization : {args.normalization}")
            print(f"   Split by      : {args.split_by}  |  "
                  f"{args.train_ratio:.2f}/{args.val_ratio:.2f}/{args.test_ratio:.2f}")
            if args.max_groups is not None:
                print(f"   Max groups    : {args.max_groups}")
            if args.max_tracks is not None:
                print(f"   Max tracks    : {args.max_tracks}")
            if args.max_frames_per_track is not None:
                print(f"   Max frames/trk: {args.max_frames_per_track}")
            print(f"   Seed          : {args.seed}")

            df, stats = build_dataset_from_temp_data_outputs(
                data_root=str(data_root),
                input_suffix=input_suffix,
                target_suffix=target_suffix,
                frame_size=args.frame_size,
                hop_size=args.hop_size,
                threshold=args.threshold,
                normalization=args.normalization,
                train_val_test_split=split_tuple,
                split_by=args.split_by,
                max_groups=args.max_groups,
                max_tracks=args.max_tracks,
                max_frames_per_track=args.max_frames_per_track,
                seed=args.seed,
            )

            _save_dataframe(df, output_path)

            # ── summary ────────────────────────────────────────────────────
            print("\n" + "─" * 52)
            print("  Dataset Build Summary")
            print("─" * 52)
            print(f"  Groups found      : {stats['num_groups']}")
            print(f"  Complete tracks   : {stats['num_complete']}")
            print(f"  Incomplete tracks : {stats['num_incomplete']}")
            print(f"  Effective groups  : {stats['effective_groups']}")
            print(f"  Effective tracks  : {stats['effective_tracks']}")
            print(f"  Frames kept       : {stats['total_frames_kept']}")
            print(f"  Frames dropped    : {stats['total_frames_dropped']}"
                  f"  (threshold={args.threshold})")
            print("  Split distribution:")
            print(f"    train : {(df['split'] == 'train').sum()}")
            print(f"    val   : {(df['split'] == 'val').sum()}")
            print(f"    test  : {(df['split'] == 'test').sum()}")
            print(f"  Columns: {', '.join(df.columns)}")
            _warn_suspicious_metadata_uniqueness(df, context="temp_data_outputs")
            print("─" * 52)

        else:  # raw_files
            print(f"\nBuilding dataset from {data_root} (source_type=raw_files)")
            print(f"   Input pattern : {args.input_pattern}")
            print(f"   Target pattern: {args.target_pattern}")
            print(f"   Frame size    : {args.frame_size}  |  Hop size: {args.hop_size}")
            print(f"   Threshold     : {args.threshold}")
            print(f"   Normalization : {args.normalization}")
            print(f"   Split ratios  : "
                  f"{args.train_ratio:.2f}/{args.val_ratio:.2f}/{args.test_ratio:.2f}")
            print(f"   Seed          : {args.seed}")
            print()

            df = build_dataset_from_raw(
                data_dir=str(data_root),
                input_pattern=args.input_pattern,
                target_pattern=args.target_pattern,
                frame_size=args.frame_size,
                hop_size=args.hop_size,
                threshold=args.threshold,
                normalization=args.normalization,
                train_val_test_split=split_tuple,
                seed=args.seed,
                sr=args.sr,
            )

            _save_dataframe(df, output_path)

            print(f"\n📈 Dataset Summary:")
            print(f"   Total frames  : {len(df)}")
            print(f"   Unique tracks : {df['track_id'].nunique()}")
            print(f"   Train frames  : {(df['split'] == 'train').sum()}")
            print(f"   Val frames    : {(df['split'] == 'val').sum()}")
            print(f"   Test frames   : {(df['split'] == 'test').sum()}")
            print(f"   Columns       : {', '.join(df.columns)}")
            _warn_suspicious_metadata_uniqueness(df, context="raw_files")

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _save_dataframe(df, output_path: Path) -> None:
    """Save DataFrame to .pkl or .parquet."""
    if output_path.suffix == ".pkl":
        with open(output_path, "wb") as f:
            pickle.dump(df, f)
        print(f"\n✅ Saved to {output_path}  (pickle, {len(df)} rows)")
    elif output_path.suffix == ".parquet":
        df.to_parquet(output_path)
        print(f"\n✅ Saved to {output_path}  (parquet, {len(df)} rows)")
    else:
        print(f"❌ Unsupported output format: {output_path.suffix}  (use .pkl or .parquet)")
        sys.exit(1)

if __name__ == "__main__":
    main()
