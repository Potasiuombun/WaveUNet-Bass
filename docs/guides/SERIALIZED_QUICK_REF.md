# Quick Reference: Serialized Datasets

Status: aligned with current `temp_data_outputs` pipeline and track-level grouped splits.

## Recommended Thesis-Grade Rebuild (temp_data_outputs)

**This is the recommended path for final experiments.**

```bash
# 1. Prepare from /mnt/f/fma_small/temp_data
python scripts/prepare_dataset.py \
    --data-root /mnt/f/fma_small/temp_data \
    --output /mnt/f/fma_small/datasets/waveunet_baseline.pkl \
    --input-suffix _reference_clipped.npy \
    --target-suffix _admm_processed.npy \
    --frame-size 1024 \
    --hop-size 512 \
    --threshold 0.5

# 2. Train
python scripts/train.py \
    --config configs/baseline.yaml \
    --dataset-file /mnt/f/fma_small/datasets/waveunet_baseline.pkl

# 3. Inspect metadata consistency
python scripts/inspect_dataset.py /mnt/f/fma_small/datasets/waveunet_baseline.pkl
```

**Expected folder structure scanned:**
```
/mnt/f/fma_small/temp_data/
    018_output/
        018031_reference_clipped.npy    ← input
        018031_admm_processed.npy       ← target
        018031_admm_processed.wav       (metadata, optional)
        018031_admm_reference_clipped.wav  (metadata, optional)
        018031_admm_rescaled.npy        (metadata, optional)
    019_output/
        ...
```

Track ID = 6-digit prefix before first `_` in filename (e.g. `018031`).
Split = grouped by `track_id` — no frame leakage across train/val/test.

For limited RAM environments, use the subset controls below.

```bash
python scripts/prepare_dataset.py \
  --data-root /mnt/f/fma_small/temp_data \
  --output /mnt/f/fma_small/datasets/waveunet_baseline_small.pkl \
  --input-suffix _reference_clipped.npy \
  --target-suffix _admm_processed.npy \
  --frame-size 1024 --hop-size 512 \
  --threshold 0.5 \
  --max-groups 2 \
  --max-tracks 120 \
  --max-frames-per-track 128
```

---

## One-Liner Quick Start (raw_files mode)

```bash
# Prepare dataset from raw files
python scripts/prepare_dataset.py --data-root data/raw --output dataset.pkl

# Train from serialized dataset (no rescanning!)
python scripts/train.py --config configs/baseline.yaml --dataset-file dataset.pkl
```

---

## Common Commands

### Prepare Datasets — temp_data_outputs mode

```bash
# Minimal (uses default suffixes)
python scripts/prepare_dataset.py \
    --data-root /mnt/f/fma_small/temp_data \
    --output /mnt/f/fma_small/datasets/waveunet_baseline.pkl \
    --source-type temp_data_outputs

# Full options
python scripts/prepare_dataset.py \
    --data-root /mnt/f/fma_small/temp_data \
    --output /mnt/f/fma_small/datasets/waveunet_baseline.pkl \
    --input-suffix _reference_clipped.npy \
    --target-suffix _admm_processed.npy \
    --frame-size 1024 --hop-size 512 \
    --threshold 0.5 \
    --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
    --seed 42
```

### Prepare Datasets — raw_files mode

```bash
# Default settings
python scripts/prepare_dataset.py --data-root data/raw --output dataset.pkl

# With threshold filtering
python scripts/prepare_dataset.py --data-root data/raw --output dataset.pkl --threshold 0.01

# With normalization
python scripts/prepare_dataset.py --data-root data/raw --output dataset.pkl --normalization peak_per_track

# Parquet format (smaller file)
python scripts/prepare_dataset.py --data-root data/raw --output dataset.parquet
```

### Training

```bash
# From serialized dataset (fast, no rescanning)
python scripts/train.py --config configs/baseline.yaml --dataset-file dataset.pkl

# From raw files (original method, slower)
python scripts/train.py --config configs/baseline.yaml

# Override data directory
python scripts/train.py --config configs/baseline.yaml --data-root data/custom_raw
```

### Inspect Datasets

```bash
# View dataset information
python scripts/inspect_dataset.py dataset.pkl

# Check validation results
python scripts/inspect_dataset.py dataset.parquet
```

---

## Decision Tree

```
Have /mnt/f/fma_small/temp_data (*_output folders)?
├─ YES  ← This is the thesis-grade path
│  └─ prepare_dataset.py --input-suffix _reference_clipped.npy ...
│     → train.py --dataset-file waveunet_baseline.pkl
│
└─ NO, have flat raw audio directory?
   ├─ First time?
   │  └─ prepare_dataset.py (raw_files mode) → train.py --dataset-file
   └─ Have a legacy DataFrame?
      └─ train.py --dataset-file (legacy auto-detected)
```

---

## File Formats Comparison

| Format | Speed | Size | Use Case |
|--------|-------|------|----------|
| .pkl | Fast | Large | Quick iteration |
| .parquet | Medium | Small | Storage, sharing |
| Raw WAV | Slow | Very Large | Flexible preprocessing |

---

## prepare_dataset.py Parameters

```
Required:
  --data-root PATH              Root folder to scan
  --output PATH                 Output file (.pkl or .parquet)

Source type (auto-detected from other flags):
  --source-type STR             'raw_files' or 'temp_data_outputs'

temp_data_outputs options:
  --input-suffix SUFFIX         Filename suffix for input NPY (default: _reference_clipped.npy)
  --target-suffix SUFFIX        Filename suffix for target NPY (default: _admm_processed.npy)
  --split-by STR                'track_id' (default) or 'group_id'

raw_files options:
  --input-pattern PATTERN       Glob for input files (default: *reference*)
  --target-pattern PATTERN      Glob for target files (default: *processed*)
  --sr INT                      Sample rate (default: 48000)

Shared framing options:
  --frame-size INT              Frame size (default: 1024)
  --hop-size INT                Hop size (default: 512)
  --threshold FLOAT             Amplitude threshold (default: 0.0)
  --normalization MODE          'none', 'peak_per_track', 'rms_per_track'
  --train-ratio FLOAT           Train ratio (default: 0.8)
  --val-ratio FLOAT             Val ratio (default: 0.1)
  --test-ratio FLOAT            Test ratio (default: 0.1)
  --seed INT                    Random seed (default: 42)

Memory-safe subset options:
  --max-groups INT              Limit number of *_output folders processed
  --max-tracks INT              Limit number of complete tracks processed
  --max-frames-per-track INT    Cap kept frames per track after filtering
```

---

## Dataset Columns (temp_data_outputs output)

| Column | Type | Description |
|--------|------|-------------|
| `input` | np.ndarray [frame_size] | Input audio frame |
| `target` | np.ndarray [frame_size] | Target audio frame |
| `track_id` | str | 6-digit track ID (e.g. `018031`) |
| `frame_index` | int | Frame position within track |
| `source_input_path` | str | Absolute path to input `.npy` |
| `source_target_path` | str | Absolute path to target `.npy` |
| `group_id` | str | Folder name (e.g. `018_output`) |
| `input_kind` | str | Kind label (`reference_clipped`) |
| `target_kind` | str | Kind label (`admm_processed`) |
| `split` | str | `train` / `val` / `test` |
| `input_peak` | float or None | Normalization value (if used) |
| `target_peak` | float or None | Normalization value (if used) |
| `processed_wav_path` | str or None | Path to `_admm_processed.wav` |
| `reference_wav_path` | str or None | Path to `_admm_reference_clipped.wav` |
| `rescaled_npy_path` | str or None | Path to `_admm_rescaled.npy` |

---

## train.py Arguments

```
Mutually exclusive (choose one):
  --dataset-file PATH           Load from serialized dataset (recommended)
  --data-root PATH              Override data directory (raw files)

Optional:
  --config PATH                 Config file (default: configs/baseline.yaml)
  --device cuda|cpu             Device override
```

Overfit run (batch size 64, NMSE-focused):

```bash
python scripts/train.py \
  --config configs/overfit_nmse.yaml \
  --dataset-file /mnt/f/fma_small/datasets/waveunet_baseline_small.pkl \
  --device cuda
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No *_output folders found" | Check --data-root path |
| "No complete tracks found" | Verify --input-suffix / --target-suffix |
| "Incomplete track ..." | Missing input or target file for that track_id |
| "No frames generated" | Lower --threshold or check audio length |
| "ModuleNotFoundError: pandas" | `pip install pandas` |
| "Dataset file not found" | Check path: `ls -lh dataset.pkl` |
| "No data found for split='train'" | Missing 'split' column in DF |

---

## New API: Python Access

```python
from src.data.serialized import (
    scan_temp_data_outputs,
    build_dataset_from_temp_data_outputs,
    get_dataset_info,
)

# Scan without building frames
track_records, stats = scan_temp_data_outputs(
    "/mnt/f/fma_small/temp_data",
    input_suffix="_reference_clipped.npy",
    target_suffix="_admm_processed.npy",
)
print(stats)  # {num_groups, num_complete, num_incomplete}

# Build full DataFrame
df, build_stats = build_dataset_from_temp_data_outputs(
    "/mnt/f/fma_small/temp_data",
    frame_size=1024,
    hop_size=512,
    threshold=0.5,
    train_val_test_split=(0.8, 0.1, 0.1),
    seed=42,
)
print(build_stats)  # {num_groups, num_complete, ..., total_frames_kept, total_frames_dropped}

# Quick info query (no full load)
info = get_dataset_info("waveunet_baseline.pkl")
print(info["num_unique_tracks"], info["split_type"])
```

---

## FAQ

**Q: Does the new mode break existing code?**
A: No — `raw_files` mode and legacy `.pkl` loading are unchanged.

**Q: How is track_id extracted?**
A: The contiguous digit prefix before the first `_` in each filename.
   `018031_reference_clipped.npy` → `track_id = "018031"`.

**Q: What if a track is missing its input or target file?**
A: A warning is printed and that track is skipped (counted as incomplete).

**Q: Can I use group_id instead of track_id for splitting?**
A: Yes, pass `--split-by group_id`. This splits at folder level — useful for debugging
   but track_id is safer for leak prevention.

**Q: How do I check the dataset is ready to train?**
```bash
python scripts/inspect_dataset.py /mnt/f/fma_small/datasets/waveunet_baseline.pkl
```

**Q: How do I convert CSV to pickle?**
```python
import pandas as pd
df = pd.read_csv("dataset.csv")
df.to_pickle("dataset.pkl")
```


## Common Commands

### Prepare Datasets

```bash
# Default settings
python scripts/prepare_dataset.py --data-root data/raw --output dataset.pkl

# With threshold filtering
python scripts/prepare_dataset.py --data-root data/raw --output dataset.pkl --threshold 0.01

# With normalization
python scripts/prepare_dataset.py --data-root data/raw --output dataset.pkl --normalization peak_per_track

# Parquet format (smaller file)
python scripts/prepare_dataset.py --data-root data/raw --output dataset.parquet
```

### Training

```bash
# From serialized dataset (fast, no rescanning)
python scripts/train.py --config configs/baseline.yaml --dataset-file dataset.pkl

# From raw files (original method, slower)
python scripts/train.py --config configs/baseline.yaml

# Override data directory
python scripts/train.py --config configs/baseline.yaml --data-root data/custom_raw
```

### Inspect Datasets

```bash
# View dataset information
python scripts/inspect_dataset.py dataset.pkl

# Check validation results
python scripts/inspect_dataset.py dataset.parquet
```

## Decision Tree

```
Do you have raw audio files?
├─ YES, first time training?
│  └─ Run: prepare_dataset.py → train.py (with --dataset-file)
│
├─ YES, training multiple times?
│  └─ Run: prepare_dataset.py (once) → train.py (multiple times with --dataset-file)
│
└─ NO, have a DataFrame already?
   ├─ Pickle or parquet format?
   │  └─ Run: train.py (with --dataset-file)
   │
   └─ CSV or other format?
      └─ Convert to pkl/parquet → train.py (with --dataset-file)
```

## File Formats Comparison

| Format | Speed | Size | Use Case |
|--------|-------|------|----------|
| .pkl | ⚡⚡⚡ Fast | Large | Quick iteration |
| .parquet | ⚡⚡ Medium | Small | Storage, sharing |
| Raw WAV | ⚡ Slow | Very Large | Flexible preprocessing |

## prepare_dataset.py Parameters

```
Required:
  --data-root PATH              Root folder with audio files
  --output PATH                 Output file (.pkl or .parquet)

Optional:
  --input-pattern PATTERN       Glob for input files (default: *reference*)
  --target-pattern PATTERN      Glob for target files (default: *processed*)
  --frame-size INT              Frame size (default: 1024)
  --hop-size INT                Hop size (default: 512)
  --threshold FLOAT             Amplitude threshold (default: 0.0)
  --normalization MODE          'none', 'peak_per_track', 'rms_per_track'
  --train-ratio FLOAT           Train ratio (default: 0.7)
  --val-ratio FLOAT             Val ratio (default: 0.15)
  --test-ratio FLOAT            Test ratio (default: 0.15)
  --seed INT                    Random seed (default: 42)
  --sr INT                      Sample rate (default: 48000)
```

## train.py New Arguments

```
Mutually exclusive (choose one):
  --dataset-file PATH           Load from serialized dataset
  --data-root PATH              Override data directory

Optional:
  --config PATH                 Config file (default: configs/baseline.yaml)
  --device cuda|cpu             Device override
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "ModuleNotFoundError: pandas" | `pip install pandas` |
| "Dataset file not found" | Check path: `ls -lh dataset.pkl` |
| "Shape mismatch" | Source files corrupted or mismatched |
| "No data found for split='train'" | Missing 'split' column in DF |

## Example Workflow

```bash
# 1. Organize data
# data/raw/
#   track_001/
#     admm_reference.wav
#     admm_processed.wav
#   track_002/
#     admm_reference.wav
#     admm_processed.wav

# 2. Prepare dataset (2-3 minutes, one time)
python scripts/prepare_dataset.py \
    --data-root data/raw \
    --output datasets/bass_v1.pkl \
    --seed 42

# Output:
# ✅ Saved dataset to datasets/bass_v1.pkl
# 📈 Total frames: 142857
#    Unique tracks: 37
#    Train: 99999, Val: 21429, Test: 21429

# 3. Inspect (optional, seconds)
python scripts/inspect_dataset.py datasets/bass_v1.pkl

# 4. Train (fast, no rescanning!)
python scripts/train.py --config configs/baseline.yaml --dataset-file datasets/bass_v1.pkl

# 5. Train again with different params (also fast!)
python scripts/train.py --config configs/baseline_deep.yaml --dataset-file datasets/bass_v1.pkl
```

## Performance Reference

Speed per 1M frames:

| Operation | Raw Files | .pkl | .parquet |
|-----------|-----------|------|----------|
| First scan | 45s | - | - |
| Load | 45s | 3s | 5s |
| Speed-up | baseline | 15x | 9x |

Size for 1M frames:

| Format | Size |
|--------|------|
| Raw WAV | 8 GB |
| .pkl | 4 GB |
| .parquet | 2.5 GB |

## Advanced: Custom DataFrames

```python
import pandas as pd
import pickle

# Create your own frames however you want
def load_my_frames():
    data = []
    for track_id, frames in my_preprocessing():
        for i, (inp, tgt) in enumerate(frames):
            data.append({
                "input": inp.numpy(),
                "target": tgt.numpy(),
                "track_id": track_id,
                "frame_index": i,
                "source_input_path": "custom",
                "source_target_path": "custom",
                "split": assign_split(track_id)
            })
    return pd.DataFrame(data)

df = load_my_frames()

# Save
with open("my_frames.pkl", "wb") as f:
    pickle.dump(df, f)

# Use in training
python scripts/train.py --config configs/baseline.yaml --dataset-file my_frames.pkl
```

## FAQ

**Q: Can I use serialized datasets with existing code?**
A: Yes, 100% backward compatible. Raw files still work exactly as before.

**Q: How do I convert CSV to pickle?**
```python
import pandas as pd
df = pd.read_csv("dataset.csv")
df.to_pickle("dataset.pkl")
```

**Q: What if my frames have different sizes?**
Make sure frame_size in prepare_dataset.py matches your data.

**Q: Can I shuffle frames across tracks?**
No (by design), but you can modify the DataFrame manually if needed.

**Q: How do I verify the dataset is correct?**
```bash
python scripts/inspect_dataset.py dataset.pkl
```

**Q: Can I resume from checkpoints with serialized data?**
Yes, the dataset is just input—checkpoints work the same way.

**Q: Performance: should I use .pkl or .parquet?**
- `.pkl`: If you have SSD and want maximum speed
- `.parquet`: If disk space matters or you want to share data
