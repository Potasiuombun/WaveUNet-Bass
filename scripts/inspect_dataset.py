#!/usr/bin/env python3
"""Utility script to inspect and validate serialized datasets."""
import argparse
import sys
from pathlib import Path
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("❌ pandas required. Install: pip install pandas")
    sys.exit(1)


def inspect_dataset(dataset_path: Path):
    """Inspect a serialized dataset file."""
    
    if not dataset_path.exists():
        print(f"❌ File not found: {dataset_path}")
        sys.exit(1)
    
    print(f"\n📋 Inspecting: {dataset_path}")
    print(f"   Size: {dataset_path.stat().st_size / (1024**3):.2f} GB")
    
    # Load
    try:
        if dataset_path.suffix == ".pkl":
            with open(dataset_path, "rb") as f:
                df = pickle.load(f)
        elif dataset_path.suffix == ".parquet":
            df = pd.read_parquet(dataset_path)
        else:
            print(f"❌ Unsupported format: {dataset_path.suffix}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        sys.exit(1)
    
    if not isinstance(df, pd.DataFrame):
        print(f"❌ Expected DataFrame, got {type(df)}")
        sys.exit(1)
    
    # Check if legacy format
    is_legacy = set(df.columns) == {"input", "target"}
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Format: {'🔶 LEGACY (input/target only)' if is_legacy else '✨ RICH (with metadata)'}")
    print(f"   Total rows: {len(df)}")
    print(f"   Columns: {', '.join(df.columns)}")
    
    # Columns analysis
    print(f"\n🔍 Column Details:")
    for col in df.columns:
        dtype = df[col].dtype
        if col in ["input", "target"]:
            # Check array properties
            sample = df[col].iloc[0]
            if hasattr(sample, 'shape'):
                shape = sample.shape
                print(f"   {col:25} {str(dtype):20} shape={shape}")
            else:
                print(f"   {col:25} {str(dtype):20} (not array-like)")
        else:
            nunique = df[col].nunique()
            print(f"   {col:25} {str(dtype):20} unique={nunique}")
    
    # Check first sample shape consistency
    try:
        sample_input = df["input"].iloc[0]
        sample_target = df["target"].iloc[0]
        print(f"\n📐 Sample Shapes (first row):")
        print(f"   input:  {sample_input.shape if hasattr(sample_input, 'shape') else 'N/A'}")
        print(f"   target: {sample_target.shape if hasattr(sample_target, 'shape') else 'N/A'}")
    except Exception as e:
        print(f"\n   Could not read sample shapes: {e}")
    
    # Splits
    if "split" in df.columns:
        print(f"\n🎯 Splits:")
        splits = df["split"].value_counts()
        for split_name in ["train", "val", "test"]:
            count = splits.get(split_name, 0)
            pct = 100 * count / len(df)
            print(f"   {split_name:8} {count:8} rows ({pct:5.1f}%)")
    else:
        print(f"\n🎯 Splits: No split column (all data unsplit)")
    
    # Track metadata
    if "track_id" in df.columns:
        n_tracks = df["track_id"].nunique()
        frames_per_track = len(df) / n_tracks
        print(f"\n🎵 Track Info:")
        print(f"   Unique track_ids: {n_tracks}")
        print(f"   Avg rows per track: {frames_per_track:.1f}")
        
        # Check if all same track (legacy pattern)
        if n_tracks == 1 and df["track_id"].iloc[0].startswith("legacy_"):
            print(f"   ⚠️  All data from single legacy track (no grouped split possible)")
    
    # Data quality checks
    print(f"\n✅ Validation:")
    checks_passed = 0
    checks_total = 0
    
    # Check required columns
    required_cols = {"input", "target"}
    missing = required_cols - set(df.columns)
    checks_total += 1
    if missing:
        print(f"   ❌ Missing required columns: {missing}")
    else:
        print(f"   ✅ Required columns (input, target) present")
        checks_passed += 1
    
    # Check shape matching
    checks_total += 1
    shape_ok = True
    shape_issues = []
    for idx in range(min(10, len(df))):  # Check first 10
        try:
            input_arr = df["input"].iloc[idx]
            target_arr = df["target"].iloc[idx]
            if len(input_arr) != len(target_arr):
                shape_ok = False
                shape_issues.append(f"Row {idx}: input len={len(input_arr)}, target len={len(target_arr)}")
        except Exception as e:
            shape_ok = False
            shape_issues.append(f"Row {idx}: {e}")
    
    if shape_ok:
        print(f"   ✅ Input/target length matching (checked {min(10, len(df))} rows)")
        checks_passed += 1
    else:
        print(f"   ❌ Shape mismatches found:")
        for issue in shape_issues[:3]:
            print(f"      {issue}")
    
    # Check for NaN
    checks_total += 1
    has_nan = False
    for col in ["input", "target"]:
        if col in df.columns:
            try:
                # Check if arrays contain NaN
                sample_has_nan = any(
                    (np.isnan(arr).any() if isinstance(arr, np.ndarray) else False)
                    for arr in df[col].iloc[:min(10, len(df))]
                )
                if sample_has_nan:
                    has_nan = True
                    break
            except:
                pass  # Couldn't check
    
    if has_nan:
        print(f"   ❌ NaN values found in input/target")
    else:
        print(f"   ✅ No NaN values (samples checked)")
        checks_passed += 1
    
    # Check metadata
    if "split" in df.columns:
        checks_total += 1
        if set(df["split"].unique()) <= {"train", "val", "test"}:
            print(f"   ✅ Split values valid")
            checks_passed += 1
        else:
            print(f"   ❌ Invalid split values: {set(df['split'].unique())}")

    # Metadata path consistency checks (rich datasets)
    if "track_id" in df.columns and df["track_id"].nunique() > 1:
        print(f"\n🧪 Metadata Path Consistency:")
        track_count = int(df["track_id"].nunique())
        path_cols = [
            "source_input_path",
            "source_target_path",
            "reference_wav_path",
            "processed_wav_path",
            "rescaled_npy_path",
        ]

        for col in path_cols:
            if col not in df.columns:
                continue

            non_null = df[col].notna()
            tracks_with_value = int(df.loc[non_null, "track_id"].nunique())
            ratio = tracks_with_value / track_count if track_count else 0.0

            # Reuse check: same path should not map to many track_ids.
            reused_count = 0
            if non_null.any():
                pairs = df.loc[non_null, ["track_id", col]].drop_duplicates()
                reused = pairs.groupby(col)["track_id"].nunique()
                reused_count = int((reused > 1).sum())

            flag = "✅"
            msg = f"{tracks_with_value}/{track_count} tracks ({ratio:.1%})"
            if tracks_with_value == 0:
                flag = "⚠️"
                msg += " - all values missing"
            elif ratio < 0.5:
                flag = "⚠️"
                msg += " - suspiciously low coverage"

            if reused_count > 0:
                flag = "❌"
                msg += f" - {reused_count} reused path(s) across tracks"

            print(f"   {flag} {col}: {msg}")
    
    print(f"\n📈 Validation: {checks_passed}/{checks_total} checks passed")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if is_legacy:
        print(f"   • This is a LEGACY pickle (only input/target columns)")
        print(f"   • No true track IDs exist for grouped splitting")
        print(f"   • For thesis-grade evaluation with proper splits, regenerate from raw files:")
        print(f"     python scripts/prepare_dataset.py --data-root <dir> --output dataset.pkl")
        print(f"   • This dataset can still be used for training (row-level splitting only)")
    else:
        if "track_id" in df.columns and df["track_id"].nunique() > 1:
            print(f"   ✓ Dataset has rich metadata - grouped split by track is available")
        else:
            print(f"   • Dataset has metadata columns but only one track ID")
    
    if checks_passed == checks_total:
        print(f"   ✓ Dataset structure looks good!")
    
    return 0 if checks_passed == checks_total else 1


def main():
    parser = argparse.ArgumentParser(description="Inspect serialized dataset")
    parser.add_argument("dataset_file", help="Path to .pkl or .parquet file")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_file)
    return inspect_dataset(dataset_path)


if __name__ == "__main__":
    sys.exit(main())
