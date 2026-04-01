"""Training callbacks."""
from typing import Optional, List


class EarlyStopper:
    """Early stopping based on validation loss."""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        restore_best: bool = True
    ):
        """Initialize early stopper.
        
        Args:
            patience: Number of epochs with no improvement to wait before stopping.
            min_delta: Minimum improvement to reset patience counter.
            restore_best: If True, can restore best model later.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        
        self.counter = 0
        self.best_loss = float("inf")
        self.best_epoch = 0
    
    def step(self, val_loss: float, epoch: int) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Validation loss for current epoch.
            epoch: Current epoch number.
            
        Returns:
            True if training should stop, else False.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        
        return False
    
    def __str__(self) -> str:
        return f"Best loss: {self.best_loss:.6f} at epoch {self.best_epoch}, counter: {self.counter}/{self.patience}"
