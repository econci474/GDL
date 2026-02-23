"""Dataset loaders for node classification benchmarks."""

import torch
from pathlib import Path
from torch_geometric.datasets import Planetoid, HeterophilousGraphDataset
from torch_geometric.transforms import NormalizeFeatures
import numpy as np

def load_dataset(name: str, root_dir: str = "data", planetoid_split: str = "public", planetoid_normalize: bool = True):
    """
    Load a node classification dataset with standard splits.
    
    Args:
        name: 'cora', 'pubmed', 'roman-empire', 'minesweeper', 'squirrel'
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

    elif name == 'squirrel':
        # Filtered Squirrel from Platonov et al. (2023) — removes duplicate nodes that cause
        # data leakage in the original WikipediaNetwork version.
        # Source: https://github.com/yandex-research/heterophilous-graphs
        import urllib.request
        from torch_geometric.data import Data

        squirrel_dir = Path(root_dir) / 'squirrel'
        npz_path = squirrel_dir / 'squirrel_filtered.npz'

        if not npz_path.exists():
            squirrel_dir.mkdir(parents=True, exist_ok=True)
            url = ('https://raw.githubusercontent.com/yandex-research/'
                   'heterophilous-graphs/main/data/squirrel_filtered.npz')
            print(f'Downloading filtered Squirrel from {url} ...')
            urllib.request.urlretrieve(url, npz_path)
            print('Done.')

        raw = np.load(npz_path)
        # npz keys: node_features (N,F), node_labels (N,), edges (E,2),
        #           train_masks (10,N), val_masks (10,N), test_masks (10,N)
        x          = torch.tensor(raw['node_features'], dtype=torch.float)
        y          = torch.tensor(raw['node_labels'],   dtype=torch.long)
        edge_index = torch.tensor(raw['edges'].T,       dtype=torch.long)  # (2, E)
        # Masks: transpose to (N, 10) to match PyG / HeterophilousGraphDataset convention
        train_mask = torch.tensor(raw['train_masks'], dtype=torch.bool).T  # (N, 10)
        val_mask   = torch.tensor(raw['val_masks'],   dtype=torch.bool).T
        test_mask  = torch.tensor(raw['test_masks'],  dtype=torch.bool).T

        data = Data(x=x, y=y, edge_index=edge_index,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        dataset_kind = 'heterophilous'

        # Expose a num_classes attribute to match the return convention
        class _SquirrelDatasetMeta:
            num_classes = int(y.max().item()) + 1
        dataset = _SquirrelDatasetMeta()

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

