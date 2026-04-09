#!/usr/bin/env python3
"""Visualize Stage-1 dataset split and losses."""

import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# ============================================================================
# 1. DATASET SPLIT OVERVIEW
# ============================================================================

print("=" * 70)
print("STAGE 1 DATASET SPLIT")
print("=" * 70)

dataset_path = Path("datasets/stage1_fma_curated_cli_norm.pkl")
df = pd.read_pickle(dataset_path)

print(f"\n📊 Total Frames: {len(df):,}")
print(f"📚 Total Unique Tracks: {df['track_id'].nunique():,}")

print("\n--- SPLIT DISTRIBUTION ---")
split_counts = df["split"].value_counts().to_dict()
for split in ["train", "val", "test"]:
    count = split_counts.get(split, 0)
    pct = 100.0 * count / len(df)
    print(f"  {split.upper():6s}: {count:5,} frames ({pct:5.1f}%)")

print("\n--- UNIQUE TRACKS PER SPLIT ---")
tracks_per_split = df.groupby("split")["track_id"].nunique().to_dict()
for split in ["train", "val", "test"]:
    count = tracks_per_split.get(split, 0)
    pct = 100.0 * count / df["track_id"].nunique()
    print(f"  {split.upper():6s}: {count:4,} tracks ({pct:5.1f}%)")

print("\n--- FRAMES PER TRACK (STATISTICS) ---")
frames_per_track = df.groupby("track_id").size()
for split in ["train", "val", "test"]:
    split_df = df[df["split"] == split]
    frames_per_track_split = split_df.groupby("track_id").size()
    print(f"\n  {split.upper()}:")
    print(f"    Mean:   {frames_per_track_split.mean():.1f} frames/track")
    print(f"    Median: {frames_per_track_split.median():.1f} frames/track")
    print(f"    Min:    {frames_per_track_split.min():,} frames/track")
    print(f"    Max:    {frames_per_track_split.max():,} frames/track")

# ============================================================================
# 2. DATASET SAMPLE EXAMPLES
# ============================================================================

print("\n" + "=" * 70)
print("SAMPLE ROWS FROM EACH SPLIT")
print("=" * 70)

for split in ["train", "val", "test"]:
    split_df = df[df["split"] == split]
    print(f"\n--- {split.upper()} (1st example) ---")
    row = split_df.iloc[0]
    print(f"  Track ID: {row['track_id']}")
    print(f"  Frame Index: {row['frame_index']}")
    print(f"  Input shape: {row['input'].shape}")
    print(f"  Target shape: {row['target'].shape}")
    print(f"  Input range: [{row['input'].min():.4f}, {row['input'].max():.4f}]")
    print(f"  Target range: [{row['target'].min():.4f}, {row['target'].max():.4f}]")
    print(f"  Normalization mode: {row['normalization_mode']}")
    print(f"  Source: {row['source_input_path']}")

# ============================================================================
# 3. LOSS FUNCTIONS
# ============================================================================

print("\n" + "=" * 70)
print("LOSS FUNCTIONS USED IN STAGE 1")
print("=" * 70)

config_path = Path("configs/stage1_fma_curated_cli_norm.yaml")
import yaml
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

loss_config = config.get("loss", {})
print("\n--- WEIGHTED LOSSES ---")
total_weight = 0.0
for loss_name, weight in loss_config.items():
    total_weight += weight
    print(f"  {loss_name:15s}: weight = {weight:.1f}")

print(f"\n  Total weight: {total_weight:.1f}")

# ============================================================================
# 4. LOSS DESCRIPTIONS
# ============================================================================

print("\n--- LOSS FUNCTION DESCRIPTIONS ---")

loss_descriptions = {
    "l1_weight": """
    📌 L1 Loss (Mean Absolute Error):
       Loss = mean(|output - target|)
       • Robust to outliers
       • Encourages sparse gradients
       • Good for perceptual audio quality
    """,
    
    "nmse_weight": """
    📌 NMSE Loss (Normalized Mean Squared Error):
       Loss = mean((output - target)²) / mean(target²)
       • Normalized by signal power
       • More sensitive to large errors (quadratic)
       • Scale-invariant through normalization
    """,
    
    "mrstft_weight": """
    📌 Multi-Resolution STFT Loss:
       Loss = sum over resolutions of (mag_loss + phase_loss)
       FFT sizes: [1024, 512, 256]
       Hop sizes: [512, 256, 128]
       • Captures spectral content at multiple scales
       • Sensitive to frequency-domain distortions
       • Particularly good for audio reconstruction
    """
}

for loss_name, description in loss_descriptions.items():
    if loss_config.get(loss_name.replace("_weight", ""), 0) > 0:
        print(description)

print("--- COMBINED LOSS ---")
print("""
    The final loss is computed as:
    
    Total Loss = L1_weight * L1 + NMSE_weight * NMSE + MRSTFT_weight * MRSTFT
    
    This combines:
    ✓ Time-domain reconstruction (L1 + NMSE)
    ✓ Frequency-domain reconstruction (MRSTFT)
    
    All losses have weight 1.0, so equal emphasis on all three objectives.
""")

# ============================================================================
# 5. TRAINING METRICS
# ============================================================================

print("=" * 70)
print("TRAINING RESULTS")
print("=" * 70)

metrics_path = Path("logs/stage1_fma_curated_cli_norm_metrics.csv")
if metrics_path.exists():
    metrics_df = pd.read_csv(metrics_path)
    print(f"\n📈 Training metrics saved: {metrics_path}")
    print(f"   Rows: {len(metrics_df)}")
    print("\n   Final epoch metrics:")
    final_row = metrics_df.iloc[-1]
    for col in ["epoch", "train_loss", "val_loss", "val_nmse", "val_mae"]:
        if col in final_row.index:
            print(f"     {col:20s}: {final_row[col]}")

# ============================================================================
# 6. VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Split distribution (frames)
ax = axes[0, 0]
splits = list(split_counts.keys())
counts = list(split_counts.values())
colors = ["#2E86AB", "#A23B72", "#F18F01"]
ax.bar(splits, counts, color=colors, alpha=0.7, edgecolor="black", linewidth=2)
ax.set_ylabel("Frame Count", fontsize=11, fontweight="bold")
ax.set_title("Frame Distribution by Split", fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(counts):
    ax.text(i, v + 200, f"{v:,}", ha="center", fontweight="bold")

# Plot 2: Track distribution
ax = axes[0, 1]
tracks = [tracks_per_split.get(s, 0) for s in ["train", "val", "test"]]
ax.bar(splits, tracks, color=colors, alpha=0.7, edgecolor="black", linewidth=2)
ax.set_ylabel("Track Count", fontsize=11, fontweight="bold")
ax.set_title("Unique Tracks by Split", fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(tracks):
    ax.text(i, v + 20, f"{v}", ha="center", fontweight="bold")

# Plot 3: Loss configuration
ax = axes[1, 0]
loss_names = [k.replace("_weight", "") for k, v in loss_config.items() if v > 0]
loss_weights = [v for k, v in loss_config.items() if v > 0]
bars = ax.barh(loss_names, loss_weights, color=["#06A77D", "#D62246", "#FF8C42"], alpha=0.7, edgecolor="black", linewidth=2)
ax.set_xlabel("Weight", fontsize=11, fontweight="bold")
ax.set_title("Loss Function Weights", fontsize=12, fontweight="bold")
ax.set_xlim([0, max(loss_weights) * 1.2])
for i, (name, weight) in enumerate(zip(loss_names, loss_weights)):
    ax.text(weight + 0.05, i, f"{weight:.1f}", va="center", fontweight="bold")

# Plot 4: Training curves (if metrics exist)
ax = axes[1, 1]
if metrics_path.exists():
    metrics_df = pd.read_csv(metrics_path)
    ax.plot(metrics_df["epoch"], metrics_df["train_loss"], "o-", label="Train Loss", linewidth=2, markersize=5)
    ax.plot(metrics_df["epoch"], metrics_df["val_loss"], "s-", label="Val Loss", linewidth=2, markersize=5)
    ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax.set_title("Loss Convergence", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, "Metrics file not found", ha="center", va="center", transform=ax.transAxes)

plt.tight_layout()
output_path = Path("logs/dataset_split_visualization.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n✅ Visualization saved: {output_path}")

print("\n" + "=" * 70)
