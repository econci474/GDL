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


def load_results_with_splits(base_path, filename='probe_results.pt'):
    """
    Load results from either single file or multiple split folders.
    Handles both homophilous (single file) and heterophilous (split folders).
    
    Args:
        base_path: Path to directory containing results
        filename: Name of the result file to load
        
    Returns:
        results: Loaded results (averaged if multiple splits)
        num_splits: Number of splits (1 for homophilous datasets)
    """
    from pathlib import Path
    
    base_path = Path(base_path)
    
    # Check for homophilous (single file)
    single_file = base_path / filename
    if single_file.exists():
        return torch.load(single_file), 1
    
    # Check for heterophilous (split folders)
    split_dirs = sorted([d for d in base_path.glob('split_*') if d.is_dir()])
    if not split_dirs:
        raise FileNotFoundError(
            f"No results found in {base_path}\n"
            f"Looked for: {filename} or split_*/{filename}"
        )
    
    # Load all splits
    all_results = []
    for split_dir in split_dirs:
        split_file = split_dir / filename
        if split_file.exists():
            all_results.append(torch.load(split_file))
    
    if not all_results:
        raise FileNotFoundError(f"No {filename} files found in split folders")
    
    # Average across splits
    averaged = average_split_results(all_results)
    return averaged, len(all_results)


def average_split_results(results_list):
    """
    Average results from multiple splits.
    
    Args:
        results_list: List of result dicts from different splits
        
    Returns:
        averaged_results: Single dict with averaged values
    """
    if len(results_list) == 1:
        return results_list[0]
    
    # Initialize with first result structure
    averaged = {}
    first = results_list[0]
    
    # Handle different result structures
    for key in first.keys():
        value = first[key]
        
        if isinstance(value, torch.Tensor):
            # Stack all splits and average
            stacked = torch.stack([r[key] for r in results_list])
            averaged[key] = stacked.mean(dim=0)
            
        elif isinstance(value, dict):
            # Recursively average nested dicts
            averaged[key] = {}
            for sub_key in value.keys():
                if isinstance(value[sub_key], torch.Tensor):
                    stacked = torch.stack([r[key][sub_key] for r in results_list])
                    averaged[key][sub_key] = stacked.mean(dim=0)
                else:
                    # Use first split's value for non-tensor data
                    averaged[key][sub_key] = value[sub_key]
                    
        elif isinstance(value, list):
            # Average each list element if it's a tensor
            averaged[key] = []
            for i in range(len(value)):
                if isinstance(value[i], torch.Tensor):
                    stacked = torch.stack([r[key][i] for r in results_list])
                    averaged[key].append(stacked.mean(dim=0))
                else:
                    averaged[key].append(value[i])
                    
        else:
            # Use first split's value for scalars and other types
            averaged[key] = value
    
    return averaged
