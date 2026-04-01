# Serialized Dataset Guide

This guide explains how to use the serialized dataset support for training without rescanning raw audio files.

## Overview

The serialized dataset feature allows you to:
1. **Prepare datasets once**: Convert raw audio files into a single `.pkl` or `.parquet` file
2. **Train faster**: Load pre-processed frames directly without file scanning at training time
3. **Legacy support**: Continue using the original raw file scanning approach if preferred
4. **No overhead**: Metadata is generated safely even for minimal DataFrames

## Quick Start

### 1. Prepare the Dataset

```bash
python scripts/prepare_dataset.py \
    --data-root data/bass_enhancement \
    --output dataset.pkl \
    --frame-size 1024 \
    --hop-size 512 \
    --threshold 0.0
```

This creates a `dataset.pkl` file with all preprocessing applied:
- Paired input/target frames
- Track-level splits (train/val/test)
- Metadata for each frame
- Optional normalization

### 2. Train from Serialized Dataset

```bash
python scripts/train.py \
    --config configs/baseline.yaml \
    --dataset-file dataset.pkl
```

Or use from parquet:

```bash
python scripts/train.py \
    --config configs/baseline.yaml \
    --dataset-file dataset.parquet
```

### 3. Continue with Raw Files (Optional)

No changes needed - if neither `--data-root` nor `--dataset-file` are provided, it uses the config:

```bash
python scripts/train.py --config configs/baseline.yaml
```

## Prepare Dataset Script

### Common Usage Patterns

**Default settings (most cases):**
```bash
python scripts/prepare_dataset.py \
    --data-root data/raw \
    --output dataset.pkl
```

**With amplitude threshold (skip silent frames):**
```bash
python scripts/prepare_dataset.py \
    --data-root data/raw \
    --output dataset.pkl \
    --threshold 0.01
```

**With per-track normalization:**
```bash
python scripts/prepare_dataset.py \
    --data-root data/raw \
    --output dataset.pkl \
    --normalization peak_per_track
```

**With custom split ratios:**
```bash
python scripts/prepare_dataset.py \
    --data-root data/raw \
    --output dataset.pkl \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

**With custom frame parameters:**
```bash
python scripts/prepare_dataset.py \
    --data-root data/raw \
    --output dataset.pkl \
    --frame-size 2048 \
    --hop-size 1024
```

### All Arguments

```
--data-root PATH           Root directory with raw audio files (required)
--output PATH              Output file path (.pkl or .parquet) (required)
--input-pattern PATTERN    Glob pattern for input files (default: *reference*)
--target-pattern PATTERN   Glob pattern for target files (default: *processed*)
--frame-size INT           Frame size in samples (default: 1024)
--hop-size INT             Hop size in samples (default: 512)
--threshold FLOAT          Amplitude threshold for filtering (default: 0.0)
--normalization MODE       'none', 'peak_per_track', 'rms_per_track' (default: none)
--train-ratio FLOAT        Training split ratio (default: 0.7)
--val-ratio FLOAT          Validation split ratio (default: 0.15)
--test-ratio FLOAT         Test split ratio (default: 0.15)
--seed INT                 Random seed for splitting (default: 42)
--sr INT                   Expected sample rate in Hz (default: 48000)
```

## DataFrame Format

### Expected Columns

**Required:**
- `input`: numpy array [frame_size] or [1, frame_size]
- `target`: numpy array [frame_size] or [1, frame_size]

**Optional (auto-generated if missing):**
- `track_id`: str - identifies which track frame came from
- `frame_index`: int - frame number within the track
- `source_input_path`: str - original input file path
- `source_target_path`: str - original target file path
- `split`: str - "train", "val", or "test"

**Auto-generated if normalization was applied:**
- `input_peak`: float - loudness value for input normalization
- `target_peak`: float - loudness value for target normalization

### What if my DataFrame is missing metadata?

The `load_serialized_dataset()` function generates synthetic metadata safely:

```python
from src.data.serialized import load_serialized_dataset

# Load with validation (default)
dataset = load_serialized_dataset("dataset.pkl", split="train", validate=True)
# Output: Loaded serialized dataset (train): 4521 frames from 12 tracks
#         (track_id, frame_index, etc. auto-generated)
```

## Training with Serialized Datasets

### From Command Line

```bash
# Use serialized dataset
python scripts/train.py --config configs/baseline.yaml --dataset-file dataset.pkl

# Override data root (raw files)
python scripts/train.py --config configs/baseline.yaml --data-root data/custom
```

### From Python Code

```python
from src.data.serialized import load_serialized_dataset

# Load a specific split for inference/evaluation
val_dataset = load_serialized_dataset(
    "dataset.pkl",
    split="val",
    batch_size=None,  # Return Dataset, not DataLoader
    validate=True
)

# Or get a DataLoader directly
val_loader = load_serialized_dataset(
    "dataset.pkl",
    split="val",
    batch_size=32,
    num_workers=2
)
```

## Workflow: From Raw Files to Training

### Step 1: Organize Your Data

```
data/bass_enhancement/
├── track_001/
│   ├── admm_reference.wav    # input
│   └── admm_processed.wav    # target
├── track_002/
│   ├── admm_reference.wav
│   └── admm_processed.wav
```

### Step 2: Prepare Dataset (One-time)

```bash
python scripts/prepare_dataset.py \
    --data-root data/bass_enhancement \
    --output datasets/bass_v1.pkl \
    --frame-size 1024 \
    --threshold 0.01 \
    --seed 42
```

Output:
```
📊 Building dataset from data/bass_enhancement/...
✅ Saved dataset to datasets/bass_v1.pkl (pickle)

📈 Dataset Summary:
   Total frames: 142857
   Unique tracks: 37
   Train frames: 99999
   Val frames: 21429
   Test frames: 21429
```

### Step 3: Train (Multiple Times, No Rescanning)

```bash
# First experiment
python scripts/train.py \
    --config configs/baseline.yaml \
    --dataset-file datasets/bass_v1.pkl

# Second experiment (same data, different hyperparams)
python scripts/train.py \
    --config configs/baseline.yaml \
    --dataset-file datasets/bass_v1.pkl \
    --device cuda
```

## Format Comparison

### Pickle (.pkl)

**Pros:**
- Fastest loading
- Python-native format
- Preserves numpy arrays directly

**Cons:**
- Larger file size
- Not human-readable
- Tied to Python version

**Use when:** Speed is critical, file size not a concern

### Parquet (.parquet)

**Pros:**
- Smaller file size (good compression)
- Cross-language support
- Can be inspected with tools like `pandas.read_parquet()`
- Better for long-term storage

**Cons:**
- Slightly slower loading
- Requires pyarrow library

**Use when:** Sharing data, long-term storage, file size matters

### Example: Inspect Parquet File

```python
import pandas as pd

df = pd.read_parquet("dataset.parquet")
print(df.info())
print(df.head())
print(f"Total size: {len(df)} frames")
print(f"Splits: {df['split'].value_counts().to_dict()}")
```

## Performance Notes

### Loading Time

- **Raw files:** ~30-60 seconds (first run, scans all files)
- **Serialized (.pkl):** ~2-5 seconds (all data loaded at once)
- **Serialized (.parquet):** ~3-8 seconds (with compression overhead)

### Disk Usage (1M frames, 1024 samples/frame)

- **Raw WAV files:** ~8 GB (uncompressed)
- **Serialized .pkl:** ~4 GB (numpy arrays + metadata)
- **Serialized .parquet:** ~2.5 GB (compressed)

## Troubleshooting

### "Dataset file not found"

```bash
# Make sure path is correct
python scripts/prepare_dataset.py --output ./datasets/dataset.pkl

# Check it exists
ls -lh datasets/dataset.pkl
```

### "No data found for split='train'"

The DataFrame is missing the `split` column. This can happen if you load a DataFrame that wasn't created by `build_dataset_from_raw()`. Use:

```python
from src.data.serialized import load_serialized_dataset

# Split=None to load all data
dataset = load_serialized_dataset("dataset.pkl", split=None)
```

### "Shape mismatch: input X vs target Y"

Input and target arrays have different lengths. This usually means:
1. Source files are mismatched
2. Frame indices got corrupted

Re-run `prepare_dataset.py` on your raw files.

### "Unsupported format: .csv"

Only `.pkl` and `.parquet` are supported. Convert your DataFrame:

```python
import pandas as pd
import pickle

# From CSV
df = pd.read_csv("dataset.csv")
with open("dataset.pkl", "wb") as f:
    pickle.dump(df, f)
```

## Advanced: Custom DataFrame Creation

If you have your own frame extraction pipeline, you can create a DataFrame and save it:

```python
import pandas as pd
import numpy as np
import pickle

# Your custom frame extraction
my_frames = load_my_frames()  # returns list of (input, target, metadata)

data = []
for input_arr, target_arr, meta in my_frames:
    data.append({
        "input": input_arr,
        "target": target_arr,
        "track_id": meta["track"],
        "frame_index": meta["idx"],
        "source_input_path": meta["input_path"],
        "source_target_path": meta["target_path"],
        "split": meta["split"]
    })

df = pd.DataFrame(data)

# Save
with open("my_dataset.pkl", "wb") as f:
    pickle.dump(df, f)

# Use in training
from src.data.serialized import load_serialized_dataset
loader = load_serialized_dataset("my_dataset.pkl", split="train", batch_size=32)
```

## Backwards Compatibility

All existing code continues to work:

```bash
# Old way (still works)
python scripts/train.py --config configs/baseline.yaml

# New way (faster, no rescanning)
python scripts/train.py --config configs/baseline.yaml --dataset-file dataset.pkl
```

The `--dataset-file` flag is optional and takes precedence over `--data-root` if both are provided.
