"""Dataset loaders for node classification benchmarks."""

import torch
from torch_geometric.datasets import Planetoid, HeterophilousGraphDataset
from torch_geometric.transforms import NormalizeFeatures
import numpy as np

def load_dataset(name: str, root_dir: str = "data", planetoid_split: str = "public", planetoid_normalize: bool = True):
    """
    Load a node classification dataset with standard splits.
    
    Args:
        name: 'cora', 'pubmed', 'roman-empire', 'minesweeper'
        root_dir: where datasets are downloaded/cached
        planetoid_split: split protocol for Planetoid datasets ("public", "full", "random")
        planetoid_normalize: whether to normalize node features for Planetoid datasets

    Returns:
        data: torch_geometric.data.Data object with:
            - data.x: [N, F] node features
            - data.y: [N] node labels
            - data.edge_index: [2, E] edge indices
            - data.train_mask, data.val_mask, data.test_mask: [N] boolean masks
    """
    name = name.lower()
    transform = NormalizeFeatures() if planetoid_normalize else None
    
    # For Cora, should the split be public or full? 
    if name == 'cora':
        dataset = Planetoid(root=root_dir, name='Cora', split=planetoid_split, transform=transform)
        data = dataset[0] #the one and only graph in this dataset
        dataset_kind = "homophilous"
        
    elif name == 'pubmed':
        dataset = Planetoid(root=root_dir, name='PubMed', split=planetoid_split, transform=transform)
        data = dataset[0] #the one and only graph in this dataset
        dataset_kind = "homophilous"
        
    elif name == 'roman-empire':
        # Use PyG's heterophilous graph dataset collection
        dataset = HeterophilousGraphDataset(root=root_dir, name='Roman-empire')
        data = dataset[0]
        dataset_kind = "heterophilous"
        
    elif name == 'minesweeper':
        dataset = HeterophilousGraphDataset(root=root_dir, name='Minesweeper')
        data = dataset[0]
        dataset_kind = "heterophilous"
            
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    # Ensure masks exist and are 1D
    for m in ["train_mask", "val_mask", "test_mask"]:
        if not hasattr(data, m) or getattr(data, m) is None:
            raise ValueError(f"Masks not found: data.{m} is missing")
    
    print(f"\nDataset: {name.capitalize()}")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Features: {data.num_features}")
    print(f"  Classes: {dataset.num_classes}")
    if data.train_mask.dim() == 1:
        print(f"  Train nodes: {int(data.train_mask.sum())}")
        print(f"  Val nodes:   {int(data.val_mask.sum())}")
        print(f"  Test nodes:  {int(data.test_mask.sum())}")
    else:
        print(f"  Train mask shape: {tuple(data.train_mask.shape)}")
        print(f"  Val mask shape:   {tuple(data.val_mask.shape)}")
        print(f"  Test mask shape:  {tuple(data.test_mask.shape)}")
    
    return data, dataset.num_classes, dataset_kind

