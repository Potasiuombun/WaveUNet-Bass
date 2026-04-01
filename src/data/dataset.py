"""PyTorch Dataset and DataLoader utilities."""
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

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


@dataclass
class FrameMetadata:
    """Metadata for a framed sample."""
    track_id: str
    frame_index: int
    source_input_path: str
    source_target_path: str
    input_peak: Optional[float] = None
    target_peak: Optional[float] = None


class PairedAudioDataset(Dataset):
    """Dataset for paired input/target audio frames.
    
    Features:
    - Supports WAV and NPY files
    - Configurable filename pairing
    - Track-level train/val/test split (no leakage)
    - Per-frame amplitude threshold filtering
    - Optional per-track normalization
    """
    
    def __init__(
        self,
        data_dir: str,
        input_pattern: str = "*reference*",
        target_pattern: str = "*processed*",
        frame_size: int = 1024,
        hop_size: int = 512,
        threshold: float = 0.0,
        normalization: Literal["none", "peak_per_track", "rms_per_track"] = "none",
        split: Literal["train", "val", "test"] = "train",
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
        sr: int = 48000
    ):
        """Initialize paired audio dataset.
        
        Args:
            data_dir: Directory containing audio files.
            input_pattern: Glob pattern for input files.
            target_pattern: Glob pattern for target files.
            frame_size: Number of samples per frame.
            hop_size: Hop size between frames.
            threshold: Keep frame only if max(abs(input)) >= threshold.
            normalization: Normalization mode ("none", "peak_per_track", "rms_per_track").
            split: Which split to use ("train", "val", "test").
            train_val_test_split: (train, val, test) ratios.
            seed: Random seed for track splitting.
            sr: Expected sample rate.
        """
        self.data_dir = data_dir
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.threshold = threshold
        self.normalization = normalization
        self.sr = sr
        
        # Find paired files
        pairs = find_pairs(
            data_dir,
            input_pattern=input_pattern,
            target_pattern=target_pattern,
            track_id_method="parent_name"
        )
        
        self.track_ids = sorted(list(pairs.keys()))
        self.pairs = pairs  # track_id -> (input_path, target_path)
        
        # Load all audio and create frames
        self.frames: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.metadata: List[FrameMetadata] = []
        
        # Collect all frames with their track IDs
        frame_indices: List[Tuple[str, int]] = []
        
        for track_id in self.track_ids:
            input_path, target_path = self.pairs[track_id]
            
            # Load audio
            input_wav = self._load_audio(input_path)
            target_wav = self._load_audio(target_path)
            
            # Check lengths match
            if input_wav.shape != target_wav.shape:
                raise ValueError(
                    f"Shape mismatch for {track_id}: "
                    f"input {input_wav.shape} vs target {target_wav.shape}"
                )
            
            # Normalize if needed
            if self.normalization != "none":
                input_wav_norm, input_meta = normalize_audio(input_wav, self.normalization)
                target_wav_norm, target_meta = normalize_audio(target_wav, self.normalization)
                input_peak = input_meta.get("peak", input_meta.get("rms"))
                target_peak = target_meta.get("peak", target_meta.get("rms"))
            else:
                input_wav_norm = input_wav
                target_wav_norm = target_wav
                input_peak = None
                target_peak = None
            
            # Frame audio
            input_frames = frame_audio(input_wav_norm, self.frame_size, self.hop_size)
            target_frames = frame_audio(target_wav_norm, self.frame_size, self.hop_size)
            
            # Filter frames by threshold
            mask = filter_frames_by_threshold(input_frames, self.threshold)
            input_frames = input_frames[mask]
            target_frames = target_frames[mask]
            
            # Record frame indices for splitting
            num_frames = len(input_frames)
            for frame_idx in range(num_frames):
                frame_indices.append((track_id, frame_idx))
                self.frames.append((input_frames[frame_idx], target_frames[frame_idx]))
                self.metadata.append(
                    FrameMetadata(
                        track_id=track_id,
                        frame_index=frame_idx,
                        source_input_path=input_path,
                        source_target_path=target_path,
                        input_peak=input_peak,
                        target_peak=target_peak
                    )
                )
        
        # Split by track
        split_ids = split_by_track(
            self.track_ids,
            train_ratio=train_val_test_split[0],
            val_ratio=train_val_test_split[1],
            test_ratio=train_val_test_split[2],
            seed=seed
        )
        
        # Filter frames by split
        filtered_indices = filter_by_split(frame_indices, split_ids, split)
        
        # Keep only frames in this split
        frame_set = set(filtered_indices)
        self.frames = [f for i, f in enumerate(self.frames) if (frame_indices[i]) in frame_set]
        self.metadata = [m for i, m in enumerate(self.metadata) if (frame_indices[i]) in frame_set]
        
        print(
            f"Loaded dataset ({split}): {len(self.frames)} frames "
            f"from {len(set(m.track_id for m in self.metadata))} tracks"
        )
    
    def _load_audio(self, path: str) -> torch.Tensor:
        """Load audio file (WAV or NPY)."""
        path_obj = Path(path)
        
        if path_obj.suffix in [".wav", ".mp3"]:
            waveform, sr = load_wav(path, sr=self.sr)
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
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single frame pair.
        
        Returns:
            Dictionary with keys:
            - "input": [1, frame_size]
            - "target": [1, frame_size]
            Plus optional metadata keys if verbose.
        """
        input_frame, target_frame = self.frames[idx]
        
        return {
            "input": input_frame,
            "target": target_frame,
            "metadata": self.metadata[idx]
        }


def create_dataloaders(
    data_dir: str,
    input_pattern: str = "*reference*",
    target_pattern: str = "*processed*",
    frame_size: int = 1024,
    hop_size: int = 512,
    threshold: float = 0.0,
    normalization: str = "none",
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
    sr: int = 48000
) -> Dict[str, DataLoader]:
    """Create train/val/test dataloaders.
    
    Args:
        data_dir: Directory containing audio files.
        input_pattern: Glob pattern for input files.
        target_pattern: Glob pattern for target files.
        frame_size: Number of samples per frame.
        hop_size: Hop size between frames.
        threshold: Amplitude threshold for frame filtering.
        normalization: Normalization mode.
        batch_size: Batch size.
        num_workers: Number of workers for data loading.
        seed: Random seed.
        sr: Expected sample rate.
        
    Returns:
        Dictionary with keys "train", "val", "test", each mapping to a DataLoader.
    """
    dataloaders = {}
    
    for split in ["train", "val", "test"]:
        dataset = PairedAudioDataset(
            data_dir=data_dir,
            input_pattern=input_pattern,
            target_pattern=target_pattern,
            frame_size=frame_size,
            hop_size=hop_size,
            threshold=threshold,
            normalization=normalization,
            split=split,
            seed=seed,
            sr=sr
        )
        
        shuffle = (split == "train")
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_frame_batch
        )
    
    return dataloaders
