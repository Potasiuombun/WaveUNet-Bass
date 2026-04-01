"""Filename pairing and naming utilities."""
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Pattern
from dataclasses import dataclass


@dataclass
class PairingRule:
    """Rule for pairing input and target files."""
    input_pattern: str  # Regex or glob pattern
    target_pattern: str
    groupby: str = "stem"  # How to group files (stem, parent_stem, etc)


def extract_track_id(filepath: str, method: str = "parent_name") -> str:
    """Extract track ID from file path.
    
    Args:
        filepath: Path to audio file.
        method: How to extract track ID:
            - "parent_name": use parent directory name
            - "stem": use filename stem
            
    Returns:
        Track ID string.
    """
    p = Path(filepath)
    if method == "parent_name":
        return p.parent.name
    elif method == "stem":
        # Remove known suffixes like _reference, _processed, etc
        stem = p.stem
        for suffix in ["_reference", "_processed", "_admm", "_clipped"]:
            stem = stem.replace(suffix, "")
        return stem
    else:
        raise ValueError(f"Unknown method: {method}")


def find_pairs(
    directory: str,
    input_pattern: str,
    target_pattern: str,
    track_id_method: str = "parent_name"
) -> Dict[str, Tuple[str, str]]:
    """Find paired input/target files in directory.
    
    Args:
        directory: Directory to search.
        input_pattern: Filename pattern for input files (e.g., "*reference*").
        target_pattern: Filename pattern for target files (e.g., "*processed*").
        track_id_method: How to extract track ID from files.
        
    Returns:
        Dictionary mapping track_id -> (input_path, target_path).
        
    Raises:
        FileNotFoundError: If files can't be paired reasonably.
    """
    dirpath = Path(directory)
    
    # Find input files
    input_files = {}  # track_id -> input_path
    for fpath in dirpath.rglob(input_pattern):
        if fpath.is_file():
            track_id = extract_track_id(str(fpath), track_id_method)
            if track_id in input_files:
                print(f"Warning: duplicate input for {track_id}: {fpath}")
            input_files[track_id] = str(fpath)
    
    # Find target files
    target_files = {}  # track_id -> target_path
    for fpath in dirpath.rglob(target_pattern):
        if fpath.is_file():
            track_id = extract_track_id(str(fpath), track_id_method)
            if track_id in target_files:
                print(f"Warning: duplicate target for {track_id}: {fpath}")
            target_files[track_id] = str(fpath)
    
    # Match pairs
    pairs = {}
    all_ids = set(input_files.keys()) | set(target_files.keys())
    
    missing_input = set(target_files.keys()) - set(input_files.keys())
    missing_target = set(input_files.keys()) - set(target_files.keys())
    
    if missing_input:
        raise FileNotFoundError(
            f"Missing input files for tracks: {missing_input}"
        )
    if missing_target:
        raise FileNotFoundError(
            f"Missing target files for tracks: {missing_target}"
        )
    
    for track_id in all_ids:
        if track_id in input_files and track_id in target_files:
            pairs[track_id] = (input_files[track_id], target_files[track_id])
    
    if not pairs:
        raise FileNotFoundError(
            f"No paired files found in {directory} matching "
            f"input_pattern={input_pattern}, target_pattern={target_pattern}"
        )
    
    return pairs
