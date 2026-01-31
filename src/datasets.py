"""Dataset loaders for node classification benchmarks."""

import torch
from torch_geometric.datasets import Planetoid, HeterophilousGraphDataset
from torch_geometric.transforms import NormalizeFeatures
import numpy as np


def load_dataset(name: str):
    """
    Load a node classification dataset with standard splits.
    
    Args:
        name: Dataset name ('Cora', 'PubMed', 'Roman-empire', 'Minesweeper')
        
    Returns:
        data: torch_geometric.data.Data object with:
            - data.x: [N, F] node features
            - data.y: [N] node labels
            - data.edge_index: [2, E] edge indices
            - data.train_mask, data.val_mask, data.test_mask: [N] boolean masks
    """
    name = name.lower()
    
    if name == 'cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())
        data = dataset[0]
        
    elif name == 'pubmed':
        dataset = Planetoid(root='/tmp/PubMed', name='PubMed', transform=NormalizeFeatures())
        data = dataset[0]
        
    elif name == 'roman-empire':
        # Use PyG's heterophilous graph dataset collection
        dataset = HeterophilousGraphDataset(root='/tmp/Roman-empire', name='Roman-empire',
                                           transform=NormalizeFeatures())
        data = dataset[0]
        # Generate splits if not provided
        if not hasattr(data, 'train_mask') or data.train_mask is None:
            data = _generate_splits(data, train_ratio=0.6, val_ratio=0.2, seed=42)
            
    elif name == 'minesweeper':
        dataset = HeterophilousGraphDataset(root='/tmp/Minesweeper', name='Minesweeper',
                                           transform=NormalizeFeatures())
        data = dataset[0]
        # Generate splits if not provided
        if not hasattr(data, 'train_mask') or data.train_mask is None:
            data = _generate_splits(data, train_ratio=0.6, val_ratio=0.2, seed=42)
            
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    # Ensure masks exist
    if not hasattr(data, 'train_mask') or data.train_mask is None:
        data = _generate_splits(data, train_ratio=0.6, val_ratio=0.2, seed=42)
    
    print(f"\nDataset: {name.capitalize()}")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Features: {data.num_features}")
    print(f"  Classes: {dataset.num_classes}")
    print(f"  Train nodes: {data.train_mask.sum().item()}")
    print(f"  Val nodes: {data.val_mask.sum().item()}")
    print(f"  Test nodes: {data.test_mask.sum().item()}")
    
    return data


def _generate_splits(data, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    Generate train/val/test splits if not provided.
    
    Args:
        data: torch_geometric.data.Data object
        train_ratio: Fraction of nodes for training
        val_ratio: Fraction of nodes for validation
        seed: Random seed for reproducibility
        
    Returns:
        data with train_mask, val_mask, test_mask added
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data


if __name__ == '__main__':
    # Test dataset loading
    print("Testing dataset loaders...")
    for dataset_name in ['Cora']:  # Start with Cora for prototype
        data = load_dataset(dataset_name)
        assert data.x is not None
        assert data.y is not None
        assert data.edge_index is not None
        assert data.train_mask is not None
        assert data.val_mask is not None
        assert data.test_mask is not None
        print(f"✓ {dataset_name} loaded successfully\n")
