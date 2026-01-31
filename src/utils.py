"""Utility functions for reproducibility and device management."""

import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Set random seeds for reproducibility across torch, numpy, and python.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_device(data, device):
    """
    Move PyTorch Geometric Data object to specified device.
    
    Args:
        data: torch_geometric.data.Data object
        device: torch device (cuda or cpu)
        
    Returns:
        data moved to device
    """
    return data.to(device)


def get_device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
