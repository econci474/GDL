"""Compute graph structural metrics including adjusted homophily."""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

from torch_geometric.utils import to_undirected
from src.datasets import load_dataset
from src.utils import set_seed


def compute_edge_homophily(edge_index, labels):
    """
    Compute edge homophily: fraction of edges connecting same-label nodes.
    
    Args:
        edge_index: [2, num_edges] undirected edge index
        labels: [num_nodes] node labels
        
    Returns:
        h_edge: Edge homophily ratio
    """
    # Ensure edge_index is undirected and deduplicated
    edge_index = to_undirected(edge_index, reduce='mean')
    
    # Get source and target labels
    src_labels = labels[edge_index[0]]
    tgt_labels = labels[edge_index[1]]
    
    # Count edges where labels match
    # Each undirected edge appears twice in edge_index, so divide by 2
    same_label_edges = (src_labels == tgt_labels).sum().item() / 2
    total_edges = edge_index.shape[1] / 2
    
    h_edge = same_label_edges / total_edges if total_edges > 0 else 0.0
    
    return h_edge, int(total_edges)


def compute_degree_weighted_label_dist(edge_index, labels, num_classes):
    """
    Compute degree-weighted label distribution.
    
    Args:
        edge_index: [2, num_edges] undirected edge index
        labels: [num_nodes] node labels
        num_classes: Number of classes
        
    Returns:
        pbar: [num_classes] degree-weighted probability distribution
    """
    # Ensure undirected
    edge_index = to_undirected(edge_index, reduce='mean')
    
    # Compute node degrees (count each edge once)
    num_nodes = labels.shape[0]
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    
    # Count degree for each node (edge_index has both directions, so each edge counted twice)
    unique_edges = edge_index.shape[1] / 2
    for node in range(num_nodes):
        degrees[node] = (edge_index[0] == node).sum() + (edge_index[1] == node).sum()
    
    # Divide by 2 since each undirected edge is represented twice
    degrees = degrees // 2
    
    # Compute D_k = sum of degrees for nodes with label k
    D = torch.zeros(num_classes, dtype=torch.long)
    for k in range(num_classes):
        mask = (labels == k)
        D[k] = degrees[mask].sum()
    
    # Compute pbar = D_k / (2 * num_edges)
    total_degree_sum = degrees.sum().item()
    pbar = D.float() / total_degree_sum if total_degree_sum > 0 else torch.zeros(num_classes)
    
    return pbar.numpy()


def compute_adjusted_homophily(edge_index, labels, num_classes):
    """
    Compute adjusted homophily per Platonov et al. 2022.
    
    h_adj = (h_edge - baseline) / (1 - baseline)
    where baseline = sum_k pbar[k]^2
    
    Args:
        edge_index: [2, num_edges] edge index
        labels: [num_nodes] node labels
        num_classes: Number of classes
        
    Returns:
        h_adj: Adjusted homophily
        h_edge: Edge homophily
        baseline: Expected homophily under random connections
    """
    h_edge, num_edges = compute_edge_homophily(edge_index, labels)
    pbar = compute_degree_weighted_label_dist(edge_index, labels, num_classes)
    
    # Compute baseline
    baseline = np.sum(pbar ** 2)
    
    # Compute h_adj
    if baseline < 1.0:
        h_adj = (h_edge - baseline) / (1 - baseline)
    else:
        # Edge case: all nodes have same label
        h_adj = 0.0
    
    return h_adj, h_edge, baseline


def get_dataset_structural_metrics(dataset_name):
    """
    Compute and return structural metrics for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        dict with: num_nodes, num_edges_undirected, num_classes, h_edge, h_adj
    """
    # Load dataset
    data = load_dataset(dataset_name)
    
    # Get graph properties
    num_nodes = data.x.shape[0]
    edge_index = data.edge_index
    labels = data.y
    num_classes = int(labels.max().item()) + 1
    
    # Compute homophily metrics
    h_adj, h_edge, baseline = compute_adjusted_homophily(edge_index, labels, num_classes)
    
    # Count undirected edges
    edge_index_undirected = to_undirected(edge_index, reduce='mean')
    num_edges_undirected = edge_index_undirected.shape[1] // 2
    
    metrics = {
        'dataset': dataset_name,
        'num_nodes': num_nodes,
        'num_edges_undirected': num_edges_undirected,
        'num_classes': num_classes,
        'h_edge': h_edge,
        'h_adj': h_adj,
        'h_baseline': baseline,
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Compute graph structural metrics including adjusted homophily')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., Cora, PubMed, Roman-empire, Minesweeper)')
    parser.add_argument('--all', action='store_true',
                       help='Compute metrics for all datasets in config')
    
    args = parser.parse_args()
    
    # Convert config module to dict
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    # Determine which datasets to process
    if args.all:
        datasets = config['datasets']
    else:
        datasets = [args.dataset]
    
    print(f"\n{'='*60}")
    print(f"Computing Structural Metrics for {len(datasets)} dataset(s)")
    print(f"{'='*60}\n")
    
    # Compute metrics for each dataset
    metrics_list = []
    for dataset_name in datasets:
        print(f"Processing {dataset_name}...")
        metrics = get_dataset_structural_metrics(dataset_name)
        metrics_list.append(metrics)
        
        print(f"  Nodes: {metrics['num_nodes']}, Edges: {metrics['num_edges_undirected']}, Classes: {metrics['num_classes']}")
        print(f"  h_edge: {metrics['h_edge']:.4f}")
        print(f"  h_adj: {metrics['h_adj']:.4f}")
        print(f"  baseline: {metrics['h_baseline']:.4f}\n")
    
    # Save to CSV
    df = pd.DataFrame(metrics_list)
    
    output_dir = Path(config['tables_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'structural_metrics.csv'
    df.to_csv(output_path, index=False)
    
    print(f"[DONE] Structural metrics saved to: {output_path}")
    print(f"\nSummary:")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
