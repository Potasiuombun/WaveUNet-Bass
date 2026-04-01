"""Training engine."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import time

from .checkpointing import Checkpoint
from .callbacks import EarlyStopper
from .metrics import nmse, mae
from ..utils.logging import CSVLogger, JSONLogger


class Trainer:
    """Main training engine."""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
        gradient_clip: Optional[float] = None,
        amp: bool = True
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train.
            criterion: Loss function.
            optimizer: Optimizer.
            device: Device to train on.
            scheduler: Optional learning rate scheduler.
            gradient_clip: Optional gradient clipping value.
            amp: Whether to use automatic mixed precision.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        self.amp = amp
        
        if amp and device != "cpu":
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training DataLoader.
            
        Returns:
            Dictionary of loss values.
        """
        self.model.train()
        total_loss = 0.0
        loss_breakdown = {}
        num_batches = 0
        
        for batch in train_loader:
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with optional AMP
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss_dict = self.criterion(outputs, targets)
                    loss = loss_dict["total"]
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict["total"]
                
                loss.backward()
                
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Accumulate loss breakdown
            for key, val in loss_dict.items():
                if key not in loss_breakdown:
                    loss_breakdown[key] = 0.0
                loss_breakdown[key] += val.item() if isinstance(val, torch.Tensor) else val
        
        # Average losses
        avg_loss = total_loss / num_batches
        for key in loss_breakdown:
            loss_breakdown[key] /= num_batches
        
        return {
            "train_loss": avg_loss,
            **{f"train_{k}": v for k, v in loss_breakdown.items()}
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation DataLoader.
            
        Returns:
            Dictionary of metric values.
        """
        self.model.eval()
        total_loss = 0.0
        loss_breakdown = {}
        total_nmse = 0.0
        total_mae = 0.0
        num_batches = 0
        
        for batch in val_loader:
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)
            
            outputs = self.model(inputs)
            loss_dict = self.criterion(outputs, targets)
            loss = loss_dict["total"]
            
            total_loss += loss.item()
            total_nmse += nmse(outputs, targets) * len(inputs)
            total_mae += mae(outputs, targets) * len(inputs)
            num_batches += 1
            
            for key, val in loss_dict.items():
                if key not in loss_breakdown:
                    loss_breakdown[key] = 0.0
                loss_breakdown[key] += val.item() if isinstance(val, torch.Tensor) else val
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_nmse = total_nmse / (num_batches * len(inputs))  # Rough average
        avg_mae = total_mae / (num_batches * len(inputs))
        
        for key in loss_breakdown:
            loss_breakdown[key] /= num_batches
        
        return {
            "val_loss": avg_loss,
            "val_nmse": avg_nmse,
            "val_mae": avg_mae,
            **{f"val_{k}": v for k, v in loss_breakdown.items()}
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        checkpoint_dir: str = "checkpoints",
        patience: int = 5,
        csv_log_path: Optional[str] = None,
        json_log_path: Optional[str] = None
    ) -> None:
        """Fit model for multiple epochs.
        
        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            num_epochs: Number of epochs to train.
            checkpoint_dir: Directory to save checkpoints.
            patience: Early stopping patience.
            csv_log_path: Optional path to save CSV logs.
            json_log_path: Optional path to save JSON logs.
        """
        early_stopper = EarlyStopper(patience=patience)
        csv_logger = CSVLogger(csv_log_path) if csv_log_path else None
        json_logger = JSONLogger(json_log_path) if json_log_path else None
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Print metrics
            print(f"\nEpoch {epoch + 1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train loss: {train_metrics['train_loss']:.6f}")
            print(f"  Val loss: {val_metrics['val_loss']:.6f}")
            print(f"  Val NMSE: {val_metrics['val_nmse']:.6f}")
            print(f"  Val MAE: {val_metrics['val_mae']:.6f}")
            
            # Log metrics
            log_dict = {
                "epoch": epoch + 1,
                **train_metrics,
                **val_metrics,
                "time": epoch_time
            }
            
            if csv_logger:
                csv_logger.log(log_dict)
            if json_logger:
                json_logger.log(log_dict)
            
            # Save latest checkpoint
            Checkpoint.save(
                self.model,
                self.optimizer,
                epoch + 1,
                log_dict,
                f"{checkpoint_dir}/latest.pth",
                self.scheduler
            )
            
            # Save best checkpoint
            if epoch == early_stopper.best_epoch:
                Checkpoint.save(
                    self.model,
                    self.optimizer,
                    epoch + 1,
                    log_dict,
                    f"{checkpoint_dir}/best.pth",
                    self.scheduler
                )
            
            # Early stopping
            if early_stopper.step(val_metrics["val_loss"], epoch + 1):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                print(f"Best validation loss: {early_stopper.best_loss:.6f} at epoch {early_stopper.best_epoch}")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time / 3600:.2f} hours")
