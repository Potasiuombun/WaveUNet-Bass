"""Logging and metrics utilities."""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class CSVLogger:
    """Simple CSV logger for metrics."""
    
    def __init__(self, filepath: str):
        """Initialize CSV logger.
        
        Args:
            filepath: Path to CSV file (will be created if doesn't exist).
        """
        self.filepath = filepath
        self.fieldnames = None
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    
    def log(self, metrics: Dict[str, Any]) -> None:
        """Log a row of metrics.
        
        Args:
            metrics: Dictionary of metric names to values.
        """
        if self.fieldnames is None:
            self.fieldnames = list(metrics.keys())
            # Write header
            with open(self.filepath, "w") as f:
                f.write(",".join(self.fieldnames) + "\n")
        
        values = [str(metrics.get(fn, "")) for fn in self.fieldnames]
        with open(self.filepath, "a") as f:
            f.write(",".join(values) + "\n")


class JSONLogger:
    """Simple JSON line logger for structured metrics."""
    
    def __init__(self, filepath: str):
        """Initialize JSON logger.
        
        Args:
            filepath: Path to JSONL file (one JSON object per line).
        """
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    
    def log(self, metrics: Dict[str, Any]) -> None:
        """Log a JSON line.
        
        Args:
            metrics: Dictionary of metric names to values.
        """
        with open(self.filepath, "a") as f:
            f.write(json.dumps(metrics) + "\n")
