"""Reproducibility utilities."""
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seed for PyTorch, NumPy, and Python's random module.
    
    Args:
        seed: Seed value for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For full reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
