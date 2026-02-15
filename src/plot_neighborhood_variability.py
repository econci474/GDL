"""
Analyze relationship between neighborhood distinctness and classification performance.

For each class, compute:
1. Neighborhood distinctness: how far neighbor class distributions deviate from uniform
2. Mean correct-class probability across layers

Plot shows whether classes with more distinctive neighborhoods achieve better performance.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def calc_entropy(probs):
    """Calculate Shannon entropy of probability distributions."""
    eps = 1e-10
    probs_safe = np.clip(probs, eps, 1.0)
    return -(probs_safe * np.log(probs_safe)).sum(axis=1)


def compute_neighborhood_class_distribution(edge_index, labels, num_classes):
    """
    Compute neighborhood class distribution for each node.
    
    Args:
        edge_index: [2, E] edge tensor
        labels: [N] node labels
        num_classes: number of classes
    
    Returns:
        neighbor_dist: [N, C] normalized neighbor class distribution for each node
    """
    num_nodes = len(labels)
    neighbor_dist = np.zeros((num_nodes, num_classes))
    
    for node_idx in range(num_nodes):
        # Find neighbors of this node
        neighbor_mask = (edge_index[0] == node_idx)
        neighbors = edge_index[1][neighbor_mask].numpy()
        
        if len(neighbors) == 0:
            # Isolated node - use uniform distribution
            neighbor_dist[node_idx] = 1.0 / num_classes
        else:
            # Count class distribution of neighbors
            neighbor_labels = labels[neighbors]
            for c in range(num_classes):
                neighbor_dist[node_idx, c] = (neighbor_labels == c).sum() / len(neighbors)
    
    return neighbor_dist


def compute_neighborhood_distinctness(neighbor_dist, num_classes):
    """
    Compute how distinct each node's neighborhood is from uniform distribution.
    
    Distinctness = sum((p_c - 1/C)^2) / C
    
    Args:
        neighbor_dist: [N, C] neighbor class distributions
        num_classes: number of classes
    
    Returns:
        distinctness: [N] distinctness score per node
    """
    uniform_dist = 1.0 / num_classes
    squared_diff = (neighbor_dist - uniform_dist) ** 2
    distinctness = squared_diff.sum(axis=1) / num_classes
    
    return distinctness


def compute_per_class_metrics(probs, labels, neighbor_dist, num_classes, split_mask):
    """
    Compute per-class neighborhood distinctness, mean correct-class probability, and mean entropy.
    
    Args:
        probs: [N_split, C] prediction probabilities (already filtered to split)
        labels: [N] true labels (full graph)
        neighbor_dist: [N, C] neighbor class distributions (full graph)
        num_classes: number of classes
        split_mask: [N] boolean mask for split nodes
    
    Returns:
        class_distinctness: [C] mean neighborhood distinctness per class
        class_correct_prob: [C] mean correct-class probability per class
        class_entropy: [C] mean entropy per class
    """
    # Compute distinctness for all nodes
    node_distinctness = compute_neighborhood_distinctness(neighbor_dist, num_classes)
    
    # Filter to split nodes
    labels_split = labels[split_mask]
    distinctness_split = node_distinctness[split_mask]
    # probs are already filtered to split nodes
    
    # Get correct-class probabilities
    correct_class_probs = probs[np.arange(len(probs)), labels_split]
    
    # Calculate entropy for all nodes
    entropy = calc_entropy(probs)
    
    # Compute per-class averages
    class_distinctness = np.zeros(num_classes)
    class_correct_prob = np.zeros(num_classes)
    class_entropy = np.zeros(num_classes)
    
    for c in range(num_classes):
        class_mask = (labels_split == c)
        if class_mask.sum() > 0:
            class_distinctness[c] = distinctness_split[class_mask].mean()
            class_correct_prob[c] = correct_class_probs[class_mask].mean()
            class_entropy[c] = entropy[class_mask].mean()
        else:
            class_distinctness[c] = np.nan
            class_correct_prob[c] = np.nan
            class_entropy[c] = np.nan
    
    return class_distinctness, class_correct_prob, class_entropy


def plot_neighborhood_variability(dataset, model, K, seed, loss_type, split='val'):
    """Generate neighborhood variability analysis plot."""
    
    # Load dataset
    from src.datasets import load_dataset as load_ds
    data_obj, num_classes, _ = load_ds(
        dataset,
        root_dir='data',
        planetoid_normalize=False,
        planetoid_split='public'
    )
    
    labels = data_obj.y.numpy()
    edge_index = data_obj.edge_index
    
    if split == 'val':
        split_mask = data_obj.val_mask.numpy()
    else:
        split_mask = data_obj.test_mask.numpy()
    
    # Compute neighborhood class distributions (constant across layers)
    print('Computing neighborhood class distributions...')
    neighbor_dist = compute_neighborhood_class_distribution(edge_index, labels, num_classes)
    
    # Load classifier outputs
    probs_path = Path(cfg.classifier_heads_dir) / loss_type / dataset / model / f'seed_{seed}' / f'K_{K}' / 'layer_probs.npz'
    if not probs_path.exists():
        raise FileNotFoundError(f"Classifier outputs not found: {probs_path}")
    
    probs_data = np.load(probs_path)
    
    # Compute metrics for each layer
    print('Computing per-class metrics...')
    all_distinctness = []
    all_correct_prob = []
    all_entropy = []
    
    for k in range(K + 1):
        probs_k = probs_data[f'{split}_probs_{k}']
        class_dist, class_prob, class_ent = compute_per_class_metrics(
            probs_k, labels, neighbor_dist, num_classes, split_mask
        )
        all_distinctness.append(class_dist)
        all_correct_prob.append(class_prob)
        all_entropy.append(class_ent)
    
    # Create plot: 2 rows x (K+1) columns
    fig, axes = plt.subplots(2, K + 1, figsize=(4 * (K + 1), 8))
    if K == 0:
        axes = axes.reshape(2, 1)
    
    fig.suptitle(f'Neighborhood Distinctness vs Classification Performance\n'
                 f'{dataset} {model} K={K} seed={seed} | {loss_type}',
                 fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.arange(num_classes))
    
    # Get class sizes for legend sorting and labeling
    labels_split = labels[split_mask]
    class_sizes = []
    for c in range(num_classes):
        n_class = (labels_split == c).sum()
        class_sizes.append((c, n_class))
    
    # Sort by size (ascending)
    class_sizes.sort(key=lambda x: x[1])
    sorted_classes = [(c, n) for c, n in class_sizes]
    
    for k in range(K + 1):
        distinctness_k = all_distinctness[k]
        correct_prob_k = all_correct_prob[k]
        entropy_k = all_entropy[k]
        
        # Top row: Correct-class probability
        ax = axes[0, k]
        for c, n_class in sorted_classes:
            if not np.isnan(distinctness_k[c]):
                label = f'C{c} (n={n_class})'
                ax.scatter(distinctness_k[c], correct_prob_k[c],
                          s=100, color=colors[c], label=label,
                          alpha=0.7, edgecolors='black', linewidth=1.5)
        
        ax.set_xlabel('Neighborhood Distinctness', fontsize=10, fontweight='bold')
        if k == 0:
            ax.set_ylabel('Mean Correct-Class\nProbability', fontsize=10, fontweight='bold')
        ax.set_title(f'Layer k={k}', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xlim([-0.02, max(0.3, np.nanmax(distinctness_k) * 1.1)])
        ax.set_ylim([0, 1.05])
        
        if k == K:
            ax.legend(fontsize=8, loc='best', ncol=1)
        
        # Bottom row: Entropy
        ax = axes[1, k]
        for c, n_class in sorted_classes:
            if not np.isnan(distinctness_k[c]):
                label = f'C{c} (n={n_class})'
                ax.scatter(distinctness_k[c], entropy_k[c],
                          s=100, color=colors[c], label=label,
                          alpha=0.7, edgecolors='black', linewidth=1.5)
        
        ax.set_xlabel('Neighborhood Distinctness', fontsize=10, fontweight='bold')
        if k == 0:
            ax.set_ylabel('Mean Entropy', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xlim([-0.02, max(0.3, np.nanmax(distinctness_k) * 1.1)])
        # Set y-axis for entropy (typically 0 to log2(num_classes))
        max_entropy = np.log(num_classes)
        ax.set_ylim([0, max_entropy * 1.05])
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(cfg.figures_dir) / dataset / model / f'K_{K}'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{dataset}_{model}_k{K}_seed{seed}_neighborhood_variability_{loss_type}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'✓ Plot saved to: {output_path}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze neighborhood variability vs classification performance')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--K', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--loss-type', type=str, required=True,
                       help='Loss type directory name (e.g., ce_plus_R_R1.0_hard)')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    
    args = parser.parse_args()
    
    print(f'\n{"="*60}')
    print(f'Neighborhood Variability Analysis: {args.model} on {args.dataset}')
    print(f'  K={args.K}, seed={args.seed}, loss_type={args.loss_type}, split={args.split}')
    print(f'{"="*60}\n')
    
    plot_neighborhood_variability(
        args.dataset, args.model, args.K, args.seed, args.loss_type, args.split
    )
    
    print(f'\n{"="*60}')
    print('✓ Analysis complete')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
