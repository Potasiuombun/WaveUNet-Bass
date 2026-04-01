"""Serialized dataset support for legacy dataframe-based training."""
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal, Union
from dataclasses import dataclass
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .dataset import FrameMetadata, PairedAudioDataset
from .naming import find_pairs, extract_track_id
from .preprocessing import (
    frame_audio, normalize_audio, filter_frames_by_threshold
)
from .splits import split_by_track, filter_by_split
from ..utils.io import load_wav, load_npy


def collate_frame_batch(batch: List[Dict]) -> Dict:
    """Custom collate function for batches with FrameMetadata.
    
    Handles cases where batch items have dictionaries with metadata objects
    that shouldn't be stacked.
    
    Args:
        batch: List of samples from dataset.
        
    Returns:
        Dictionary with batched tensors and list of metadata.
    """
    # Assume all items have "input", "target", "metadata"
    inputs = torch.stack([b["input"] for b in batch])
    targets = torch.stack([b["target"] for b in batch])
    metadata = [b["metadata"] for b in batch]
    
    return {
        "input": inputs,
        "target": targets,
        "metadata": metadata
    }


def _normalize_frame_shape(frame: torch.Tensor) -> torch.Tensor:
    """Normalize frame shape to (1, T).
    
    Handles multiple input formats:
    - (T,) -> (1, T)
    - (T, 1) -> (1, T)
    - (1, T) -> (1, T) [no change]
    
    Args:
        frame: Input tensor
        
    Returns:
        Tensor with shape (1, T)
        
    Raises:
        ValueError: If shape is not recognized
    """
    if frame.ndim == 1:
        # (T,) -> (1, T)
        return frame.unsqueeze(0)
    elif frame.ndim == 2:
        if frame.shape[0] == 1:
            # (1, T) -> (1, T)
            return frame
        elif frame.shape[1] == 1:
            # (T, 1) -> (1, T)
            return frame.squeeze(1).unsqueeze(0)
        else:
            raise ValueError(
                f"Cannot normalize shape {frame.shape}. "
                f"Expected (T,), (1, T), or (T, 1)"
            )
    else:
        raise ValueError(
            f"Cannot normalize shape {frame.shape}. "
            f"Expected 1D or 2D tensor"
        )


def _validate_dataframe_format(df: pd.DataFrame, validate_all_rows: bool = False) -> None:
    """Validate DataFrame format and contents.
    
    Args:
        df: DataFrame to validate
        validate_all_rows: If True, validate all rows (slower)
        
    Raises:
        ValueError: If validation fails
    """
    # Check first row
    row = df.iloc[0]
    input_arr = row["input"]
    target_arr = row["target"]
    
    # Check types
    if not isinstance(input_arr, (np.ndarray, torch.Tensor)):
        raise ValueError(
            f"Row 0: 'input' has type {type(input_arr).__name__}, "
            f"expected numpy.ndarray or torch.Tensor"
        )
    if not isinstance(target_arr, (np.ndarray, torch.Tensor)):
        raise ValueError(
            f"Row 0: 'target' has type {type(target_arr).__name__}, "
            f"expected numpy.ndarray or torch.Tensor"
        )
    
    # Check shapes
    try:
        input_len = len(input_arr)
        target_len = len(target_arr)
    except (TypeError, AttributeError):
        raise ValueError(
            f"Row 0: 'input' or 'target' not array-like "
            f"(input type: {type(input_arr).__name__}, target type: {type(target_arr).__name__})"
        )
    
    if input_len != target_len:
        raise ValueError(
            f"Row 0: shape mismatch - input {input_arr.shape} vs target {target_arr.shape}"
        )
    
    # Check dimensionality
    if input_arr.ndim == 1:
        pass  # OK: (T,)
    elif input_arr.ndim == 2:
        if input_arr.shape[0] == 1 or input_arr.shape[1] == 1:
            pass  # OK: (1, T) or (T, 1)
        else:
            raise ValueError(
                f"Row 0: input shape {input_arr.shape} not supported. "
                f"Expected (T,), (1, T), or (T, 1)"
            )
    else:
        raise ValueError(
            f"Row 0: input shape {input_arr.shape} not supported. "
            f"Expected 1D or 2D array"
        )
    
    # Check dtype (can be converted to float)
    if hasattr(input_arr, 'dtype'):
        if not np.issubdtype(input_arr.dtype, np.floating) and \
           not np.issubdtype(input_arr.dtype, np.integer):
            raise ValueError(
                f"Row 0: input dtype {input_arr.dtype} not numeric. "
                f"Expected floating or integer type"
            )
    
    # Optionally validate all rows
    if validate_all_rows:
        for idx in range(1, len(df)):
            row = df.iloc[idx]
            input_arr = row["input"]
            target_arr = row["target"]
            
            if not isinstance(input_arr, (np.ndarray, torch.Tensor)):
                raise ValueError(
                    f"Row {idx}: 'input' has type {type(input_arr).__name__}, "
                    f"expected numpy.ndarray or torch.Tensor"
                )
            
            try:
                if len(input_arr) != len(target_arr):
                    raise ValueError(
                        f"Row {idx}: shape mismatch - input {input_arr.shape} vs target {target_arr.shape}"
                    )
            except TypeError:
                raise ValueError(
                    f"Row {idx}: 'input' or 'target' not array-like"
                )


class SerializedDataset(Dataset):
    """Dataset loaded from serialized pandas DataFrame.
    
    Supports both .pkl and .parquet formats, including legacy DataFrames
    with only input/target columns (no metadata).
    
    Required DataFrame columns:
    - input: numpy array or tensor, shapes (T,), (1, T), or (T, 1)
    - target: numpy array or tensor, shapes (T,), (1, T), or (T, 1)
    
    Optional columns (auto-generated if missing):
    - track_id: str (for grouped splitting)
    - frame_index: int
    - source_input_path: str
    - source_target_path: str
    - split: str ("train", "val", "test")
    
    Legacy datasets with only input/target columns:
    - Treated as single-track data (no grouped split available)
    - track_id set to constant "legacy_track_000000"
    - Grouped splitting by track not available (use row-level split)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        split: Optional[Literal["train", "val", "test"]] = None,
        validate: bool = True,
        validate_all_rows: bool = False
    ):
        """Initialize from DataFrame.
        
        Args:
            df: pandas DataFrame with input/target columns.
            split: Optional split filter ("train", "val", "test").
            validate: Whether to validate shapes/dtypes on load (first row + random sample).
            validate_all_rows: If True, validate all rows (slower but thorough).
            
        Raises:
            ValueError: If required columns missing or validation fails.
        """
        # Validate required columns
        required_cols = {"input", "target"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        if len(df) == 0:
            raise ValueError("DataFrame is empty")
        
        # Detect if this is a legacy dataset (only input/target columns)
        self.is_legacy_format = set(df.columns) == {"input", "target"}
        
        # Filter by split if provided
        if split is not None:
            if "split" not in df.columns and not self.is_legacy_format:
                raise ValueError(
                    f"Cannot filter by split='{split}' without 'split' column in DataFrame"
                )
            if "split" in df.columns:
                df = df[df["split"] == split].reset_index(drop=True)
                if len(df) == 0:
                    raise ValueError(f"No data found for split='{split}'")
        
        # Validate shapes and types
        if validate:
            _validate_dataframe_format(df, validate_all_rows)
        
        self.df = df
        self.split = split
        
        # Generate synthetic metadata for missing columns
        # For legacy format: all rows belong to same "track" (no grouped split)
        if "track_id" not in self.df.columns:
            if self.is_legacy_format:
                self.df["track_id"] = "legacy_track_000000"
            else:
                self.df["track_id"] = [f"track_{i}" for i in range(len(self.df))]
        
        if "frame_index" not in self.df.columns:
            self.df["frame_index"] = range(len(self.df))
        
        if "source_input_path" not in self.df.columns:
            self.df["source_input_path"] = "<serialized>"
        
        if "source_target_path" not in self.df.columns:
            self.df["source_target_path"] = "<serialized>"
        
        # Print dataset info
        n_unique_tracks = self.df["track_id"].nunique()
        dataset_type = "LEGACY (no metadata)" if self.is_legacy_format else "RICH (with metadata)"
        split_info = f"split={split}" if split else "no split"
        
        print(
            f"Loaded serialized dataset [{dataset_type}] ({split_info}): "
            f"{len(self.df)} frames"
            f"{f' from {n_unique_tracks} tracks' if n_unique_tracks > 1 else ''}"
        )
        
        if self.is_legacy_format:
            print(
                "  ⚠️  Legacy format detected: no true track metadata exists.\n"
                "  Grouped train/val/test split not available.\n"
                "  Row-level deterministic splitting will be used if needed."
            )
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single frame pair.
        
        Handles various input shapes:
        - (T,) -> normalized to (1, T)
        - (T, 1) -> normalized to (1, T)
        - (1, T) -> kept as (1, T)
        
        Returns:
            Dictionary with:
            - "input": torch.Tensor [1, T] or [T] (normalized)
            - "target": torch.Tensor [1, T] or [T] (normalized)
            - "metadata": FrameMetadata
        """
        row = self.df.iloc[idx]
        
        # Convert to torch tensors
        input_arr = row["input"]
        target_arr = row["target"]
        
        # Convert numpy to torch
        if isinstance(input_arr, np.ndarray):
            input_tensor = torch.from_numpy(input_arr).float()
        else:
            input_tensor = torch.as_tensor(input_arr).float()
        
        if isinstance(target_arr, np.ndarray):
            target_tensor = torch.from_numpy(target_arr).float()
        else:
            target_tensor = torch.as_tensor(target_arr).float()
        
        # Normalize shapes to (1, T)
        input_tensor = _normalize_frame_shape(input_tensor)
        target_tensor = _normalize_frame_shape(target_tensor)
        
        # Create metadata
        source_input_path = row.get("source_input_path", "<serialized>")
        if isinstance(source_input_path, (list, np.ndarray)):
            source_input_path = str(source_input_path)
        source_target_path = row.get("source_target_path", "<serialized>")
        if isinstance(source_target_path, (list, np.ndarray)):
            source_target_path = str(source_target_path)
        
        metadata = FrameMetadata(
            track_id=str(row["track_id"]),
            frame_index=int(row["frame_index"]),
            source_input_path=source_input_path,
            source_target_path=source_target_path,
            input_peak=float(row["input_peak"]) if "input_peak" in row and row["input_peak"] is not None else None,
            target_peak=float(row["target_peak"]) if "target_peak" in row and row["target_peak"] is not None else None,
        )
        
        return {
            "input": input_tensor,
            "target": target_tensor,
            "metadata": metadata
        }
    
    @property
    def has_grouped_split_capability(self) -> bool:
        """Check if this dataset supports grouped split by track.
        
        Returns False for legacy datasets where all rows belong to a single track.
        """
        return not self.is_legacy_format and self.df["track_id"].nunique() > 1


def load_serialized_dataset(
    dataset_path: Union[str, Path],
    split: Optional[Literal["train", "val", "test"]] = None,
    batch_size: int = 32,
    num_workers: int = 2,
    validate: bool = True,
    validate_all_rows: bool = False,
) -> Union[SerializedDataset, DataLoader]:
    """Load serialized dataset from .pkl or .parquet.
    
    Automatically detects and handles:
    - Legacy datasets (only input/target columns)
    - Rich datasets (with metadata columns)
    - Various array shapes: (T,), (1, T), (T, 1)
    
    Args:
        dataset_path: Path to .pkl or .parquet file.
        split: Optional split filter ("train", "val", "test").
        batch_size: If provided, returns DataLoader; else returns Dataset.
        num_workers: Number of workers for DataLoader.
        validate: Whether to validate on load (first row + sample).
        validate_all_rows: If True, validate every row (slow but thorough).
        
    Returns:
        SerializedDataset or DataLoader depending on batch_size.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If validation fails.
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Load DataFrame
    if dataset_path.suffix == ".pkl":
        with open(dataset_path, "rb") as f:
            df = pickle.load(f)
    elif dataset_path.suffix == ".parquet":
        df = pd.read_parquet(dataset_path)
    else:
        raise ValueError(f"Unsupported format: {dataset_path.suffix}. Use .pkl or .parquet")
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected DataFrame, got {type(df)}")
    
    # Create dataset
    dataset = SerializedDataset(
        df,
        split=split,
        validate=validate,
        validate_all_rows=validate_all_rows
    )
    
    # Return DataLoader if batch_size provided
    if batch_size is not None:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_frame_batch
        )
    
    return dataset


def get_dataset_info(dataset_path: Union[str, Path]) -> Dict:
    """Get metadata about a serialized dataset without loading it fully.
    
    Args:
        dataset_path: Path to .pkl or .parquet file.
        
    Returns:
        Dictionary with info:
        - is_legacy: bool
        - num_frames: int
        - has_split_column: bool
        - has_track_id_column: bool
        - num_unique_tracks: int
        - columns: list
        - suggested_split_type: str ("grouped" or "row-level")
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Load DataFrame
    if dataset_path.suffix == ".pkl":
        with open(dataset_path, "rb") as f:
            df = pickle.load(f)
    elif dataset_path.suffix == ".parquet":
        df = pd.read_parquet(dataset_path)
    else:
        raise ValueError(f"Unsupported format: {dataset_path.suffix}")
    
    is_legacy = set(df.columns) == {"input", "target"}
    has_split = "split" in df.columns
    has_track_id = "track_id" in df.columns
    num_unique_tracks = df["track_id"].nunique() if has_track_id else 1
    
    info = {
        "is_legacy": is_legacy,
        "num_frames": len(df),
        "has_split_column": has_split,
        "has_track_id_column": has_track_id,
        "num_unique_tracks": num_unique_tracks,
        "columns": list(df.columns),
        "split_type": "grouped (by track)" if (has_track_id and num_unique_tracks > 1) else "row-level (deterministic)",
    }
    
    return info


def build_dataset_from_raw(
    data_dir: Union[str, Path],
    input_pattern: str = "*reference*",
    target_pattern: str = "*processed*",
    frame_size: int = 1024,
    hop_size: int = 512,
    threshold: float = 0.0,
    normalization: Literal["none", "peak_per_track", "rms_per_track"] = "none",
    train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    sr: int = 48000,
) -> pd.DataFrame:
    """Build dataset DataFrame from raw audio files.
    
    Creates a DataFrame with all frames and metadata, including split assignment.
    
    Args:
        data_dir: Directory containing audio files.
        input_pattern: Glob pattern for input files.
        target_pattern: Glob pattern for target files.
        frame_size: Number of samples per frame.
        hop_size: Hop size between frames.
        threshold: Keep frame only if max(abs(input)) >= threshold.
        normalization: Normalization mode.
        train_val_test_split: (train, val, test) ratios.
        seed: Random seed for track splitting.
        sr: Expected sample rate.
        
    Returns:
        DataFrame with columns:
        - input: numpy array [frame_size]
        - target: numpy array [frame_size]
        - track_id: str
        - frame_index: int
        - source_input_path: str
        - source_target_path: str
        - split: str ("train", "val", or "test")
        - input_peak: float (if normalization != "none")
        - target_peak: float (if normalization != "none")
    """
    data_dir = Path(data_dir)
    
    # Build raw dataset using existing PairedAudioDataset infrastructure
    # We'll load all splits together, then split ourselves
    dataset_obj = PairedAudioDataset(
        data_dir=str(data_dir),
        input_pattern=input_pattern,
        target_pattern=target_pattern,
        frame_size=frame_size,
        hop_size=hop_size,
        threshold=threshold,
        normalization=normalization,
        split="train",  # Dummy split (we'll reassign)
        train_val_test_split=train_val_test_split,
        seed=seed,
        sr=sr
    )
    
    # Actually, we need to rebuild without filtering by split
    # Let's do it manually instead
    
    from .naming import find_pairs
    
    # Find paired files
    pairs = find_pairs(
        str(data_dir),
        input_pattern=input_pattern,
        target_pattern=target_pattern,
        track_id_method="parent_name"
    )
    
    track_ids = sorted(list(pairs.keys()))
    
    # Build split mapping
    split_ids = split_by_track(
        track_ids,
        train_ratio=train_val_test_split[0],
        val_ratio=train_val_test_split[1],
        test_ratio=train_val_test_split[2],
        seed=seed
    )
    
    # Collect all frames
    data_rows = []
    
    for track_id in track_ids:
        input_path, target_path = pairs[track_id]
        
        # Determine split
        split = None
        for split_name, split_ids_list in split_ids.items():
            if track_id in split_ids_list:
                split = split_name
                break
        
        # Load audio
        input_wav = _load_audio(input_path, sr)
        target_wav = _load_audio(target_path, sr)
        
        # Check lengths match
        if input_wav.shape != target_wav.shape:
            raise ValueError(
                f"Shape mismatch for {track_id}: "
                f"input {input_wav.shape} vs target {target_wav.shape}"
            )
        
        # Normalize if needed
        input_peak, target_peak = None, None
        if normalization != "none":
            input_wav, input_meta = normalize_audio(input_wav, normalization)
            target_wav, target_meta = normalize_audio(target_wav, normalization)
            input_peak = input_meta.get("peak", input_meta.get("rms"))
            target_peak = target_meta.get("peak", target_meta.get("rms"))
        
        # Frame audio
        input_frames = frame_audio(input_wav, frame_size, hop_size)
        target_frames = frame_audio(target_wav, frame_size, hop_size)
        
        # Filter by threshold
        mask = filter_frames_by_threshold(input_frames, threshold)
        input_frames = input_frames[mask]
        target_frames = target_frames[mask]
        
        # Add frames to data
        for frame_idx, (input_frame, target_frame) in enumerate(
            zip(input_frames, target_frames)
        ):
            data_rows.append({
                "input": input_frame.numpy() if isinstance(input_frame, torch.Tensor) else input_frame,
                "target": target_frame.numpy() if isinstance(target_frame, torch.Tensor) else target_frame,
                "track_id": track_id,
                "frame_index": frame_idx,
                "source_input_path": str(input_path),
                "source_target_path": str(target_path),
                "split": split,
                "input_peak": input_peak,
                "target_peak": target_peak,
            })
    
    df = pd.DataFrame(data_rows)
    
    print(
        f"Built dataset: {len(df)} frames from {len(track_ids)} tracks "
        f"(train: {(df['split']=='train').sum()}, "
        f"val: {(df['split']=='val').sum()}, "
        f"test: {(df['split']=='test').sum()})"
    )
    
    return df


def _load_audio(path: Union[str, Path], sr: int = 48000) -> torch.Tensor:
    """Load audio file (WAV or NPY)."""
    path_obj = Path(path)
    
    if path_obj.suffix in [".wav", ".mp3"]:
        waveform, sr_loaded = load_wav(path, sr=sr)
        return waveform.squeeze(0)  # [T]
    elif path_obj.suffix == ".npy":
        arr = load_npy(path)
        if arr.ndim == 2:
            arr = arr.squeeze(0)  # If [1, T], make [T]
        elif arr.ndim == 1:
            pass  # Already [T]
        else:
            raise ValueError(f"Unexpected shape for NPY: {arr.shape}")
        return torch.from_numpy(arr).float()
    else:
        raise ValueError(f"Unsupported format: {path_obj.suffix}")


def _load_npy_1d(path: Union[str, Path]) -> np.ndarray:
    """Load a NumPy .npy file as a float32 array.

    Handles three cases:

    1. Already 1-D ``(T,)``      — returned as-is (raw waveform).
    2. Pre-framed ``(N, F)``     — returned as-is; rows are individual frames.
    3. Squeezable channel dim:
       ``(1, T)`` → ``(T,)``;   ``(T, 1)`` → ``(T,)``.

    Returns:
        - ``(T,)`` float32 array for raw waveform files.
        - ``(N, F)`` float32 array for pre-framed files (rows = frames).
    """
    arr = np.load(str(path)).astype(np.float32)
    if arr.ndim == 1:
        return arr                      # (T,) – raw waveform
    elif arr.ndim == 2:
        if arr.shape[0] == 1:           # (1, T) → (T,)
            return arr.squeeze(0)
        elif arr.shape[1] == 1:         # (T, 1) → (T,)
            return arr.squeeze(1)
        else:
            return arr                  # (N, F) – pre-framed matrix
    else:
        raise ValueError(
            f"Unexpected NPY dimensionality {arr.ndim} (shape {arr.shape}) at {path}"
        )


def scan_temp_data_outputs(
    data_root: Union[str, Path],
    input_suffix: str = "_reference_clipped.npy",
    target_suffix: str = "_admm_processed.npy",
    aux_suffixes: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict], Dict]:
    """Scan ``*_output`` folders under *data_root* for per-track file bundles.

    Expected directory layout::

        data_root/
            018_output/
                018031_reference_clipped.npy
                018031_admm_processed.npy
                018031_admm_processed.wav         (optional)
                018031_admm_reference_clipped.wav (optional)
                018031_admm_rescaled.npy          (optional)

    Track ID is the contiguous digit prefix before the first ``_`` in each
    filename (e.g. ``"018031"`` from ``018031_reference_clipped.npy``).

    Args:
        data_root: Root directory containing ``*_output`` sub-folders.
        input_suffix: Filename suffix that identifies the input file for each
            track (e.g. ``"_reference_clipped.npy"``).
        target_suffix: Filename suffix that identifies the target file for each
            track (e.g. ``"_admm_processed.npy"``).
        aux_suffixes: Optional ``{column_name: file_suffix}`` mapping for
            auxiliary files attached as metadata paths.  When *None*, defaults
            to::

                {
                    "processed_wav_path":  "_admm_processed.wav",
                    "reference_wav_path":  "_admm_reference_clipped.wav",
                    "rescaled_npy_path":   "_admm_rescaled.npy",
                }

    Returns:
        ``(track_records, stats)`` where:

        - ``track_records`` — list of dicts, one per *complete* track, with
          keys: ``track_id``, ``group_id``, ``input_path``, ``target_path``,
          plus one key per entry in *aux_suffixes* (``None`` if not found).
        - ``stats`` — ``{"num_groups": int, "num_complete": int,
          "num_incomplete": int}``.
    """
    _aux: Dict[str, str] = aux_suffixes if aux_suffixes is not None else {
        "processed_wav_path": "_admm_processed.wav",
        "reference_wav_path": "_admm_reference_clipped.wav",
        "rescaled_npy_path":  "_admm_rescaled.npy",
    }

    data_root = Path(data_root)
    output_folders = sorted(
        [d for d in data_root.iterdir() if d.is_dir() and d.name.endswith("_output")]
    )

    if not output_folders:
        raise FileNotFoundError(
            f"No *_output folders found under {data_root}. "
            "Verify --data-root points to the correct directory."
        )

    num_complete = 0
    num_incomplete = 0
    track_records: List[Dict] = []

    for folder in output_folders:
        group_id = folder.name
        # Accumulate files per track_id within this group folder
        track_files: Dict[str, Dict] = {}

        for fpath in sorted(folder.iterdir()):
            if not fpath.is_file():
                continue
            # Derive track_id: leading digit run before the first '_'
            parts = fpath.name.split("_", 1)
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            track_id = parts[0]

            if track_id not in track_files:
                track_files[track_id] = {
                    "input_path": None,
                    "target_path": None,
                    **{col: None for col in _aux},
                }

            fname = fpath.name
            if fname.endswith(input_suffix):
                track_files[track_id]["input_path"] = fpath
            elif fname.endswith(target_suffix):
                track_files[track_id]["target_path"] = fpath
            else:
                for col, suffix in _aux.items():
                    if fname.endswith(suffix):
                        track_files[track_id][col] = fpath
                        break

        for track_id, files in sorted(track_files.items()):
            missing = []
            if files["input_path"] is None:
                missing.append(f"input (suffix={input_suffix})")
            if files["target_path"] is None:
                missing.append(f"target (suffix={target_suffix})")

            if missing:
                print(
                    f"  ⚠ Incomplete track {track_id} in {group_id}: "
                    f"missing {', '.join(missing)}"
                )
                num_incomplete += 1
                continue

            record: Dict = {
                "track_id":   track_id,
                "group_id":   group_id,
                "input_path":  str(files["input_path"]),
                "target_path": str(files["target_path"]),
            }
            for col in _aux:
                record[col] = str(files[col]) if files[col] is not None else None

            track_records.append(record)
            num_complete += 1

    stats = {
        "num_groups":    len(output_folders),
        "num_complete":  num_complete,
        "num_incomplete": num_incomplete,
    }

    return track_records, stats


def build_dataset_from_temp_data_outputs(
    data_root: Union[str, Path],
    input_suffix: str = "_reference_clipped.npy",
    target_suffix: str = "_admm_processed.npy",
    frame_size: int = 1024,
    hop_size: int = 512,
    threshold: float = 0.0,
    normalization: Literal["none", "peak_per_track", "rms_per_track"] = "none",
    train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    split_by: Literal["track_id", "group_id"] = "track_id",
    max_groups: Optional[int] = None,
    max_tracks: Optional[int] = None,
    max_frames_per_track: Optional[int] = None,
    seed: int = 42,
    aux_suffixes: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Build a serialized DataFrame from a ``temp_data_outputs`` folder structure.

    Scans all ``*_output`` sub-directories under *data_root*, pairs
    input/target ``.npy`` files by the leading digit track-ID prefix in each
    filename, frames them, applies amplitude-threshold filtering, and assigns
    a grouped train/val/test split.

    Expected directory layout::

        data_root/
            018_output/
                018031_reference_clipped.npy
                018031_admm_processed.npy
                018031_admm_processed.wav         (optional)
                018031_admm_reference_clipped.wav (optional)
                018031_admm_rescaled.npy          (optional)
            019_output/
                ...

    Args:
        data_root: Root directory containing ``*_output`` folders.
        input_suffix: Filename suffix that identifies the input file per track.
            Default: ``"_reference_clipped.npy"``.
        target_suffix: Filename suffix that identifies the target file per track.
            Default: ``"_admm_processed.npy"``.
        frame_size: Frame length in samples.
        hop_size: Hop size between consecutive frames in samples.
        threshold: Keep a frame only when ``max(|input_frame|) >= threshold``.
        normalization: Per-track normalization applied before framing
            (``"none"``, ``"peak_per_track"``, or ``"rms_per_track"``).
        train_val_test_split: ``(train, val, test)`` ratios, must sum to 1.0.
        split_by: Grouping key for the split — ``"track_id"`` (default,
            recommended; no frame leakage across tracks) or ``"group_id"``
            (folder-level split, useful for quick debugging).
        max_groups: Optional cap on number of ``*_output`` folders (smallest
            lexical names first) to include.
        max_tracks: Optional cap on total number of complete tracks.
        max_frames_per_track: Optional cap on kept frames per track after
            threshold filtering.
        seed: Random seed for reproducible splitting.
        aux_suffixes: Optional ``{column_name: suffix}`` mapping for auxiliary
            files attached as metadata paths.  Defaults to processed_wav,
            reference_wav, and rescaled_npy.

    Returns:
        ``(df, stats)`` where:

        ``df`` is a :class:`pandas.DataFrame` with columns::

            input, target,
            track_id, frame_index,
            source_input_path, source_target_path,
            group_id, input_kind, target_kind,
            split, input_peak, target_peak,
            [processed_wav_path, reference_wav_path, rescaled_npy_path]

        ``stats`` is a dict::

            {num_groups, num_complete, num_incomplete,
             total_frames_kept, total_frames_dropped}
    """
    _aux: Dict[str, str] = aux_suffixes if aux_suffixes is not None else {
        "processed_wav_path": "_admm_processed.wav",
        "reference_wav_path": "_admm_reference_clipped.wav",
        "rescaled_npy_path":  "_admm_rescaled.npy",
    }

    print(f"\nScanning {data_root} for *_output folders...")
    track_records, scan_stats = scan_temp_data_outputs(
        data_root, input_suffix, target_suffix, _aux
    )

    if not track_records:
        raise RuntimeError(
            f"No complete tracks found under {data_root}. "
            "Verify --input-suffix and --target-suffix match your filenames."
        )

    print(
        f"  {scan_stats['num_groups']} groups  |  "
        f"{scan_stats['num_complete']} complete tracks  |  "
        f"{scan_stats['num_incomplete']} incomplete tracks"
    )

    # Safety check: each auxiliary path should map to at most one track.
    # If a path is seen for multiple track_ids, metadata assignment is suspect.
    for col in _aux.keys():
        path_to_tracks: Dict[str, set] = {}
        for rec in track_records:
            p = rec.get(col)
            if p is None:
                continue
            path_to_tracks.setdefault(str(p), set()).add(str(rec["track_id"]))

        reused = {p: tids for p, tids in path_to_tracks.items() if len(tids) > 1}
        if reused:
            print(
                f"  ⚠ Metadata consistency warning: column '{col}' has "
                f"{len(reused)} path(s) reused across multiple track_ids"
            )

    # Optional dataset size controls to avoid OOM for very large corpora.
    if max_groups is not None:
        allowed_groups = sorted({r["group_id"] for r in track_records})[:max_groups]
        allowed_group_set = set(allowed_groups)
        track_records = [r for r in track_records if r["group_id"] in allowed_group_set]
        print(f"  Applying max_groups={max_groups}: {len(allowed_groups)} groups retained")

    if max_tracks is not None:
        track_records = sorted(track_records, key=lambda r: (r["group_id"], r["track_id"]))[:max_tracks]
        print(f"  Applying max_tracks={max_tracks}: {len(track_records)} tracks retained")

    # Build grouped split mapping
    if split_by == "track_id":
        split_keys = sorted({r["track_id"] for r in track_records})
    else:
        split_keys = sorted({r["group_id"] for r in track_records})

    split_ids = split_by_track(
        split_keys,
        train_ratio=train_val_test_split[0],
        val_ratio=train_val_test_split[1],
        test_ratio=train_val_test_split[2],
        seed=seed,
    )

    key_to_split: Dict[str, str] = {}
    for sname, keys in split_ids.items():
        for k in keys:
            key_to_split[k] = sname

    # Derive human-readable kind labels from file suffixes
    # e.g. "_reference_clipped.npy" → "reference_clipped"
    input_kind = input_suffix.lstrip("_").rsplit(".", 1)[0]
    target_kind = target_suffix.lstrip("_").rsplit(".", 1)[0]
    aux_cols = list(_aux.keys())

    data_rows: List[Dict] = []
    total_kept = 0
    total_dropped = 0

    for record in track_records:
        track_id = record["track_id"]
        group_id = record["group_id"]
        split_key = track_id if split_by == "track_id" else group_id
        split = key_to_split.get(split_key, "train")

        # Load NPY arrays as 1-D float32
        try:
            input_arr = _load_npy_1d(record["input_path"])
            target_arr = _load_npy_1d(record["target_path"])
        except Exception as exc:
            print(f"  ⚠ Skipping track {track_id}: load error — {exc}")
            continue

        if len(input_arr) != len(target_arr):
            print(
                f"  ⚠ Skipping track {track_id}: "
                f"length mismatch input={len(input_arr)} target={len(target_arr)}"
            )
            continue

        input_wav_t = torch.from_numpy(input_arr).float()
        target_wav_t = torch.from_numpy(target_arr).float()

        # Per-track normalization (optional)
        input_peak, target_peak = None, None
        if normalization != "none":
            input_wav_t, inp_meta = normalize_audio(input_wav_t, normalization)
            target_wav_t, tgt_meta = normalize_audio(target_wav_t, normalization)
            input_peak = inp_meta.get("peak", inp_meta.get("rms"))
            target_peak = tgt_meta.get("peak", tgt_meta.get("rms"))

        # Build [N, 1, F] frame tensors — branch on pre-framed vs raw waveform
        if input_arr.ndim == 2:
            # Already framed: (N, F) → [N, 1, F]
            if input_arr.shape[1] != frame_size:
                print(
                    f"  ⚠ Skipping track {track_id}: "
                    f"pre-framed column count {input_arr.shape[1]} != frame_size {frame_size}"
                )
                continue
            input_frames  = input_wav_t.unsqueeze(1)   # [N, 1, F]
            target_frames = target_wav_t.unsqueeze(1)
        else:
            # Raw waveform: run frame_audio
            try:
                input_frames  = frame_audio(input_wav_t,  frame_size, hop_size)
                target_frames = frame_audio(target_wav_t, frame_size, hop_size)
            except ValueError as exc:
                print(f"  ⚠ Skipping track {track_id}: framing error — {exc}")
                continue

        # Amplitude threshold filter
        n_before = input_frames.shape[0]
        mask = filter_frames_by_threshold(input_frames, threshold)
        input_frames = input_frames[mask]
        target_frames = target_frames[mask]
        n_after = input_frames.shape[0]

        total_kept += n_after
        total_dropped += n_before - n_after

        # Squeeze channel dim → [N, frame_size] for storage
        input_np = input_frames.squeeze(1).numpy()
        target_np = target_frames.squeeze(1).numpy()

        if max_frames_per_track is not None and n_after > max_frames_per_track:
            input_np = input_np[:max_frames_per_track]
            target_np = target_np[:max_frames_per_track]
            n_after = max_frames_per_track

        for frame_idx in range(n_after):
            row: Dict = {
                "input":              input_np[frame_idx],
                "target":             target_np[frame_idx],
                "track_id":            track_id,
                "frame_index":         frame_idx,
                "source_input_path":   record["input_path"],
                "source_target_path":  record["target_path"],
                "group_id":            group_id,
                "input_kind":          input_kind,
                "target_kind":         target_kind,
                "split":               split,
                "input_peak":          input_peak,
                "target_peak":         target_peak,
            }
            for col in aux_cols:
                row[col] = record.get(col)
            data_rows.append(row)

    if not data_rows:
        raise RuntimeError(
            "No frames generated. All tracks may be below threshold or too short. "
            f"frame_size={frame_size}, threshold={threshold}"
        )

    df = pd.DataFrame(data_rows)

    effective_groups = int(df["group_id"].nunique()) if "group_id" in df.columns else 0
    effective_tracks = int(df["track_id"].nunique()) if "track_id" in df.columns else 0

    build_stats: Dict = {
        "num_groups":           scan_stats["num_groups"],
        "num_complete":         scan_stats["num_complete"],
        "num_incomplete":       scan_stats["num_incomplete"],
        "effective_groups":     effective_groups,
        "effective_tracks":     effective_tracks,
        "max_groups":           max_groups,
        "max_tracks":           max_tracks,
        "max_frames_per_track": max_frames_per_track,
        "total_frames_kept":    total_kept,
        "total_frames_dropped": total_dropped,
    }

    return df, build_stats
