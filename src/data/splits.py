"""Track-level dataset splitting."""
from typing import Dict, List, Tuple
import numpy as np


def split_by_track(
    track_ids: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[str]]:
    """Split track IDs into train/val/test without leakage.
    
    Args:
        track_ids: List of unique track IDs.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.
        
    Returns:
        Dictionary with keys "train", "val", "test", each mapping to a list of track IDs.
        
    Raises:
        ValueError: If ratios don't sum to 1 or splits are empty.
    """
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    rng = np.random.RandomState(seed)
    track_ids_shuffled = rng.permutation(track_ids)
    
    n = len(track_ids_shuffled)
    n_train = max(1, int(np.round(n * train_ratio)))
    n_val = max(1, int(np.round(n * val_ratio)))
    n_test = n - n_train - n_val
    
    if n_test < 0:
        raise ValueError("Too few tracks for given split ratios")
    
    train_ids = list(track_ids_shuffled[:n_train])
    val_ids = list(track_ids_shuffled[n_train : n_train + n_val])
    test_ids = list(track_ids_shuffled[n_train + n_val:])
    
    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }


def filter_by_split(
    frame_indices: List[Tuple[str, int]],  # [(track_id, frame_idx), ...]
    split_ids: Dict[str, List[str]],
    split_name: str
) -> List[Tuple[str, int]]:
    """Filter frame indices by split.
    
    Args:
        frame_indices: List of (track_id, frame_index) tuples.
        split_ids: Dictionary with split names as keys.
        split_name: Which split to filter by ("train", "val", or "test").
        
    Returns:
        Filtered list of frame indices.
    """
    allowed_ids = set(split_ids[split_name])
    return [(track_id, frame_idx) for track_id, frame_idx in frame_indices
            if track_id in allowed_ids]
