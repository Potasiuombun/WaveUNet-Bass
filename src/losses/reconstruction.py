"""Reconstruction losses for audio."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    """L1 (MAE) loss."""
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(output, target)


class MSELoss(nn.Module):
    """Mean Squared Error loss."""
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(output, target)


class NMSELoss(nn.Module):
    """Normalised Mean Squared Error loss.
    
    NMSE = MSE / (mean(target^2) + eps)
    """
    
    def __init__(self, eps: float = 1e-8):
        """Initialize NMSE loss.
        
        Args:
            eps: Small value to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(output, target)
        normalizer = torch.mean(target ** 2) + self.eps
        return mse / normalizer
