"""Combined loss functions."""
import torch
import torch.nn as nn
from typing import Dict, Optional

from .reconstruction import L1Loss, MSELoss, NMSELoss
from .spectral import MultiResolutionSTFTLoss


class CombinedLoss(nn.Module):
    """Weighted combination of multiple losses."""
    
    def __init__(self, loss_weights: Dict[str, float]):
        """Initialize combined loss.
        
        Args:
            loss_weights: Dictionary mapping loss names to weights.
                         Supported: "l1", "mse", "nmse", "mrstft"
        """
        super().__init__()
        
        self.loss_weights = loss_weights
        self.losses = {}
        
        if "l1" in loss_weights and loss_weights["l1"] > 0:
            self.losses["l1"] = L1Loss()
        
        if "mse" in loss_weights and loss_weights["mse"] > 0:
            self.losses["mse"] = MSELoss()
        
        if "nmse" in loss_weights and loss_weights["nmse"] > 0:
            self.losses["nmse"] = NMSELoss()
        
        if "mrstft" in loss_weights and loss_weights["mrstft"] > 0:
            self.losses["mrstft"] = MultiResolutionSTFTLoss(
                fft_sizes=[1024, 512, 256],
                hop_sizes=[512, 256, 128],
                window="hann"
            )
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute combined loss.
        
        Args:
            output: Predicted waveform [batch, 1, T].
            target: Target waveform [batch, 1, T].
            
        Returns:
            Dictionary with keys for each loss and "total".
        """
        loss_values = {}
        total_loss = torch.tensor(0.0, device=output.device)
        
        for loss_name, loss_fn in self.losses.items():
            weight = self.loss_weights.get(loss_name, 0.0)
            if weight > 0:
                loss_val = loss_fn(output, target)
                loss_values[loss_name] = loss_val.detach()
                total_loss = total_loss + weight * loss_val
        
        loss_values["total"] = total_loss
        return loss_values


def create_combined_loss(
    l1_weight: float = 1.0,
    nmse_weight: float = 1.0,
    mrstft_weight: float = 1.0
) -> CombinedLoss:
    """Create a default combined loss for loudness enhancement.
    
    Args:
        l1_weight: Weight for L1 loss.
        nmse_weight: Weight for NMSE loss.
        mrstft_weight: Weight for multi-resolution STFT loss.
        
    Returns:
        CombinedLoss instance.
    """
    weights = {
        "l1": l1_weight,
        "nmse": nmse_weight,
        "mrstft": mrstft_weight
    }
    return CombinedLoss(weights)
