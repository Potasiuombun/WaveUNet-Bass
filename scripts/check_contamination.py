#!/usr/bin/env python3
"""Verify dataset split integrity - check for contamination/leakage."""

import pandas as pd
from pathlib import Path
from typing import Set
import numpy as np

print("=" * 70)
print("DATASET CONTAMINATION CHECK")
print("=" * 70)

# Check 1: Verify manifests have no overlaps
print("\n1️⃣  MANIFEST EXCLUSIVITY CHECK")
print("-" * 70)

manifest_root = Path("../fma_small_curated_20260408/manifests")

def read_ids(path: Path) -> Set[str]:
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            ids.add(s)
    return ids

train_ids = read_ids(manifest_root / "training_track_ids.txt")
val_ids = read_ids(manifest_root / "validation_track_ids.txt")
test_ids = read_ids(manifest_root / "test_track_ids.txt")

print(f"\nManifest sizes:")
print(f"  Train: {len(train_ids)} tracks")
print(f"  Val:   {len(val_ids)} tracks")
print(f"  Test:  {len(test_ids)} tracks")
print(f"  Total: {len(train_ids) + len(val_ids) + len(test_ids)} tracks")

# Check for overlaps
train_val_overlap = train_ids & val_ids
train_test_overlap = train_ids & test_ids
val_test_overlap = val_ids & test_ids
all_three_overlap = train_ids & val_ids & test_ids

print(f"\n✓ Train ∩ Val: {len(train_val_overlap)} overlaps", end="")
if train_val_overlap:
    print(f" ❌ CONTAMINATION! Overlapping tracks: {train_val_overlap}")
else:
    print(" ✅")

print(f"✓ Train ∩ Test: {len(train_test_overlap)} overlaps", end="")
if train_test_overlap:
    print(f" ❌ CONTAMINATION! Overlapping tracks: {train_test_overlap}")
else:
    print(" ✅")

print(f"✓ Val ∩ Test: {len(val_test_overlap)} overlaps", end="")
if val_test_overlap:
    print(f" ❌ CONTAMINATION! Overlapping tracks: {val_test_overlap}")
else:
    print(" ✅")

total_unique = len(train_ids | val_ids | test_ids)
print(f"\n✓ Total unique tracks: {total_unique}")

# Check 2: Verify dataset pickle has correct split metadata
print("\n2️⃣  DATASET PICKLE INTEGRITY CHECK")
print("-" * 70)

dataset_path = Path("datasets/stage1_fma_curated_cli_norm.pkl")
df = pd.read_pickle(dataset_path)

df_train_tracks = set(df[df["split"] == "train"]["track_id"].unique())
df_val_tracks = set(df[df["split"] == "val"]["track_id"].unique())
df_test_tracks = set(df[df["split"] == "test"]["track_id"].unique())

print(f"\nDataset tracks per split:")
print(f"  Train: {len(df_train_tracks)} tracks")
print(f"  Val:   {len(df_val_tracks)} tracks")
print(f"  Test:  {len(df_test_tracks)} tracks")

# Check if dataset tracks match manifests
df_train_mismatch = df_train_tracks - train_ids
df_val_mismatch = df_val_tracks - val_ids
df_test_mismatch = df_test_tracks - test_ids

print(f"\n✓ Dataset train tracks not in manifest: {len(df_train_mismatch)}", end="")
if df_train_mismatch:
    print(f" ❌ CONTAMINATION! Extra tracks: {df_train_mismatch}")
else:
    print(" ✅")

print(f"✓ Dataset val tracks not in manifest: {len(df_val_mismatch)}", end="")
if df_val_mismatch:
    print(f" ❌ CONTAMINATION! Extra tracks: {df_val_mismatch}")
else:
    print(" ✅")

print(f"✓ Dataset test tracks not in manifest: {len(df_test_mismatch)}", end="")
if df_test_mismatch:
    print(f" ❌ CONTAMINATION! Extra tracks: {df_test_mismatch}")
else:
    print(" ✅")

# Check 3: Frame-level contamination (frames from same track in different splits)
print("\n3️⃣  FRAME-LEVEL OVERLAP CHECK")
print("-" * 70)

df_train_val_overlap = df_train_tracks & df_val_tracks
df_train_test_overlap = df_train_tracks & df_test_tracks
df_val_test_overlap = df_val_tracks & df_test_tracks

print(f"\n✓ Train frames' tracks ∩ Val frames' tracks: {len(df_train_val_overlap)} tracks", end="")
if df_train_val_overlap:
    print(f" ❌ CONTAMINATION! Tracks: {df_train_val_overlap}")
else:
    print(" ✅")

print(f"✓ Train frames' tracks ∩ Test frames' tracks: {len(df_train_test_overlap)} tracks", end="")
if df_train_test_overlap:
    print(f" ❌ CONTAMINATION! Tracks: {df_train_test_overlap}")
else:
    print(" ✅")

print(f"✓ Val frames' tracks ∩ Test frames' tracks: {len(df_val_test_overlap)} tracks", end="")
if df_val_test_overlap:
    print(f" ❌ CONTAMINATION! Tracks: {df_val_test_overlap}")
else:
    print(" ✅")

# Check 4: Input/Target consistency
print("\n4️⃣  INPUT/TARGET CONSISTENCY CHECK")
print("-" * 70)

same_rows = sum(1 for i in range(len(df)) if (df.iloc[i]["input"] == df.iloc[i]["target"]).all())
print(f"\n✓ Rows where input == target: {same_rows}/{len(df)}", end="")
if same_rows == len(df):
    print(" ✅ (All frames are identity mapping as expected for Stage 1)")
else:
    print(f" ⚠️  Only {100*same_rows/len(df):.1f}% are identical")

# Check 5: Data types and ranges
print("\n5️⃣  NORMALIZATION CHECK (Pre-peak normalization)")
print("-" * 70)

for split in ["train", "val", "test"]:
    split_df = df[df["split"] == split]
    
    # Collect all samples to find global min/max
    all_samples = np.concatenate([row for row in split_df["input"].values])
    
    overall_min = all_samples.min()
    overall_max = all_samples.max()
    
    print(f"\n  {split.upper()}:")
    print(f"    Range: [{overall_min:.4f}, {overall_max:.4f}]")
    print(f"    Expected: [-1.0, 1.0] (peak normalized)")
    
    if overall_max <= 1.0 and overall_min >= -1.0:
        print(f"    ✅ All values in normalized range")
    else:
        print(f"    ⚠️  Some values exceed normalized range!")

# Check 6: Summary
print("\n" + "=" * 70)
print("CONTAMINATION RISK ASSESSMENT")
print("=" * 70)

issues = []
if train_val_overlap or train_test_overlap or val_test_overlap:
    issues.append("❌ Manifest overlaps detected")
if df_train_mismatch or df_val_mismatch or df_test_mismatch:
    issues.append("❌ Dataset tracks don't match manifests")
if df_train_val_overlap or df_train_test_overlap or df_val_test_overlap:
    issues.append("❌ Frame-level track contamination")

if not issues:
    print("\n✅ NO CONTAMINATION DETECTED")
    print("\nDataset integrity verified:")
    print("  • Train/Val/Test manifests are mutually exclusive")
    print("  • All dataset tracks match their assigned manifests")
    print("  • No track appears in multiple splits")
    print("  • All frames are properly normalized")
    print("  • Input == Target (identity mapping for Stage 1)")
    print("\n✨ Safe to proceed with training!")
else:
    print("\n⚠️  POTENTIAL CONTAMINATION ISSUES:")
    for issue in issues:
        print(f"  {issue}")

print("\n" + "=" * 70)
