# Legacy Dataset Support

## Overview

The serialized dataset implementation now includes first-class support for legacy DataFrame formats with only `input` and `target` columns and no metadata. This guide explains:

- What constitutes a "legacy" dataset
- How they are automatically detected and handled
- Limitations and workarounds
- Recommendations for thesis-grade evaluation

## What is a Legacy Dataset?

A legacy (or "framed") dataset is a pickled pandas DataFrame with exactly two columns:

```python
df = pd.DataFrame({
    "input": [np.array of shape (1024,) or (1024, 1), ...],
    "target": [np.array of shape (1024,) or (1024, 1), ...],
})
df.to_pickle("dataset.pkl")
```

**Characteristics:**
- Only `input` and `target` columns
- No track IDs or metadata
- All rows treated as coming from the same audio track
- No information about train/val/test splits

**Common origins:**
- From older processing pipelines
- Direct frame extraction without metadata tracking
- Minimal information to conserve disk space


## Automatic Detection

When you load a legacy dataset, it is automatically detected:

```bash
$ python scripts/train.py --config configs/baseline.yaml --dataset-file dataset.pkl

Dataset Info:
  Type: 🔶 LEGACY (input/target only)
  Total frames: 35627
  Unique tracks: 1
  Has split column: False
  Available split type: row-level
  ⚠️  NOTE: Legacy format detected (no track metadata).
     Grouped split not available. Row-level splitting will be used.
```

## Shape Support

Legacy datasets may have frames in any of these shapes:

| Shape | Description | Auto-converted |
|-------|-------------|---|
| `(T,)` | 1D array, e.g. `(1024,)` | → `(1, 1024)` in DataLoader |
| `(1, T)` | 2D array [1, time] e.g. `(1, 1024)` | → `(1, 1024)` (no change) |
| `(T, 1)` | 2D array [time, 1] e.g. `(1024, 1)` | → `(1, 1024)` |

All shapes are normalized to `(1, T)` in the DataLoader for training.

```python
# Your frame could have any of these shapes
frame1 = np.random.randn(1024)        # (1024,)
frame2 = np.random.randn(1, 1024)     # (1, 1024)
frame3 = np.random.randn(1024, 1)     # (1024, 1)

# When used in training, all become (batch_size, 1, 1024)
```

## Auto-Generated Metadata

When loading a legacy dataset, the system safely generates:

```python
track_id = "legacy_track_000000"      # All rows have the same constant ID
frame_index = 0, 1, 2, ...            # Sequential frame indices
source_input_path = "<serialized>"    # Indicates data is from pickle, not raw file
source_target_path = "<serialized>"
split = None                           # No pre-assigned split
```

These values are **never invented per-row**. All rows belong to a single logical track.

## Splitting Behavior

### Legacy Dataset (no metadata)

**Default behavior (if no split column exists):**

```bash
$ python scripts/train.py --config configs/baseline.yaml --dataset-file dataset.pkl
```

- If `split` column exists in DataFrame: uses it
- If `split` column missing (legacy): uses **row-level deterministic splitting**

**Row-level splitting is:**
- Deterministic based on seed
- Not grouped by track
- Can cause **data leakage** at frame boundaries
- ⚠️ Not recommended for thesis-grade evaluation

### Rich Dataset (with metadata)

If your DataFrame has `track_id` and multiple unique values:

```bash
$ python scripts/train.py --config configs/baseline.yaml --dataset-file dataset.pkl
```

- Uses **grouped split by track**
- No data leakage between splits
- Proper train/val/test separation

## Inspection

To see if a dataset is legacy:

```bash
$ python scripts/inspect_dataset.py dataset.pkl

Format: 🔶 LEGACY (input/target only)
Total rows: 35627
Columns: input, target
Unique track_ids: 1
⚠️  All data from single legacy track (no grouped split possible)
```

## Workarounds for Legacy Datasets

### Option 1: Regenerate from Raw Files (Recommended)

The best approach for thesis-grade evaluation:

```bash
# Original raw files:
# data/bass_enhancement/
#   track_001/
#     admm_reference.wav
#     admm_processed.wav

# Regenerate with rich metadata:
python scripts/prepare_dataset.py \
    --data-root data/bass_enhancement \
    --output dataset_v2.pkl

# New dataset has:
# - True track_id values
# - Grouped train/val/test splits
# - No data leakage
```

**This dataset can now be used with confidence.**

### Option 2: Add Split Column to Legacy Dataset

If you have the original split assignment:

```python
import pandas as pd
import pickle

# Load legacy pickle
df = pd.read_pickle("dataset.pkl")

# Add split column (your assignment logic)
df["split"] = assign_splits_somehow(df)  # your function

# Save as new pickle
df.to_pickle("dataset_with_splits.pkl")
```

Then use:
```bash
python scripts/train.py --config configs/baseline.yaml --dataset-file dataset_with_splits.pkl
```

### Option 3: Use Raw Files Instead

If you have the original raw files (`.wav` or `.npy`):

```bash
# Skip the legacy pickle entirely
python scripts/train.py --config configs/baseline.yaml --data-root data/raw
```

This uses automatic track-level splitting (no leakage).

## Training with Legacy Datasets

**What works:**
- ✅ Can train on legacy data with row-level splitting
- ✅ Training loop runs normally
- ✅ Loss functions work the same
- ✅ Checkpoints and inference work the same

**What to be aware of:**
- ⚠️ Train/val/test splits may not be truly separate (data leakage at frame boundaries)
- ⚠️ Cannot use grouped splitting by track
- ⚠️ Results may not be comparable to tracked datasets

**Example training:**

```bash
# Will work but with row-level splitting
python scripts/train.py \
    --config configs/baseline.yaml \
    --dataset-file legacy_dataset.pkl
```

Output:
```
Dataset Info:
  Type: 🔶 LEGACY (input/target only)
  ⚠️  NOTE: Legacy format detected (no track metadata).
     Grouped split not available. Row-level splitting will be used.
     For leak-free train/val/test splits, regenerate from raw files
```

## Recommendations

### For Development / Prototyping
Legacy datasets are fine:
- Fast iteration
- Quick experiments
- Feature development

### For Thesis / Evaluation
Use rich datasets:
```bash
# Regenerate from raw files
python scripts/prepare_dataset.py --data-root data/raw --output dataset.pkl

# Verify it's rich
python scripts/inspect_dataset.py dataset.pkl
# → Shows: ✨ RICH (with metadata)

# Train with confidence
python scripts/train.py --config configs/baseline.yaml --dataset-file dataset.pkl
```

## Storage Comparison

For 1M frames with frame_size=1024:

| Format | Size | Properties |
|--------|------|-----------|
| Legacy pickle (input/target only) | 4 GB | Minimal metadata |
| Rich pickle (with track_id, split, etc.) | 4.2 GB | Full metadata |
| Raw WAV files | 8 GB | Maximum flexibility |

The overhead of rich metadata is minimal.

## Technical Details

### Memory Layout

Legacy arrays can be:
- C-contiguous (normal numpy default)
- Fortran-contiguous (row-major)
- Non-contiguous (views, slices)

All are properly converted:
```python
# Your legacy array could be non-contiguous
arr = np.random.randn(1024, 10)[:, 0]  # Non-contiguous view

# Still converts correctly
tensor = torch.from_numpy(arr).float()
# Result is valid, may trigger ascontiguousarray internally
```

### Validation

When loading a legacy dataset with `validate=True`:

1. **First row validation** (fast): Check types, shapes, dtypes
2. **All rows validation** (slower, optional): Check structure of every row

Example (first row only):
```bash
# Fast validation (default)
python scripts/train.py --dataset-file dataset.pkl
# Checks first row immediately, all others on access
```

### Custom Metadata

If you need to add custom metadata to a legacy dataset:

```python
import pandas as pd

df = pd.read_pickle("dataset.pkl")

# Add custom columns
df["custom_loudness"] = [compute_loudness(x) for x in df["input"]]
df["speaker_id"] = [get_speaker(idx) for idx in df.index]

# Save
df.to_pickle("dataset_enriched.pkl")
```

These are preserved and accessible via metadata:

```python
from src.data.serialized import load_serialized_dataset

loader = load_serialized_dataset("dataset_enriched.pkl", batch_size=32)

for batch in loader:
    metadata_list = batch["metadata"]
    for meta in metadata_list:
        # Access via the metadata object or original columns
        print(f"Track: {meta.track_id}, Index: {meta.frame_index}")
```

## Frequently Asked Questions

**Q: Can I use a legacy dataset for final evaluation?**
A: Not recommended without regeneration. You can't guarantee train/val test separation.

**Q: My legacy dataset has 35K frames but only 1 track. Is that OK?**
A: It means all 35K frames are from the same source. Row-level splitting will be used.

**Q: Why not just add track_id to my legacy pickle?**
A: Because without knowing the original track source, any track ID would be arbitrary.

**Q: Can I convert legacy to rich offline?**
A: Yes! If you have the original raw files, use `prepare_dataset.py`.

**Q: What if I lost the raw files?**
A: You can add a `split` column manually if you have documentation of the original split.

**Q: Why does legacy_track_000000 matter?**
A: It signals "this came from a legacy pickle" and prevents accidental grouped splitting.

## See Also

- [SERIALIZED_DATASET_GUIDE.md](SERIALIZED_DATASET_GUIDE.md) - Full dataset guide
- [SERIALIZED_QUICK_REF.md](SERIALIZED_QUICK_REF.md) - Quick reference
- `scripts/prepare_dataset.py --help` - Create rich datasets from raw files
- `scripts/inspect_dataset.py --help` - Inspect any dataset
