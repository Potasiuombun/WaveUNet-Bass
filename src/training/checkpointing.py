"""Checkpointing utilities."""
import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional


class Checkpoint:
    """Checkpoint saver/loader."""
    
    @staticmethod
    def save(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        filepath: str,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    ) -> None:
        """Save checkpoint.
        
        Args:
            model: Model to save.
            optimizer: Optimizer state to save.
            epoch: Current epoch number.
            metrics: Dictionary of metrics.
            filepath: Path to save checkpoint.
            scheduler: Optional scheduler to save.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        state_dict = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics
        }
        
        if scheduler is not None:
            state_dict["scheduler"] = scheduler.state_dict()
        
        torch.save(state_dict, filepath)
        print(f"Saved checkpoint: {filepath}")
    
    @staticmethod
    def load(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        filepath: str,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Load checkpoint.
        
        Args:
            model: Model to load into.
            optimizer: Optimizer to load.
            filepath: Path to checkpoint.
            scheduler: Optional scheduler to load.
            device: Device to load on.
            
        Returns:
            Dictionary with "epoch" and "metrics".
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        state = torch.load(filepath, map_location=device)
        
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        
        if scheduler is not None and "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        
        print(f"Loaded checkpoint: {filepath}")
        
        return {
            "epoch": state.get("epoch", 0),
            "metrics": state.get("metrics", {})
        }
