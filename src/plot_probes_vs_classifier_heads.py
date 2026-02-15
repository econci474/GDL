"""Visualization comparing linear probes vs classifier heads for each layer."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def load_probe_results(dataset, model, K, seed):
    """Load linear probe results from .npz file."""
    # Load from arrays directory which contains per-node probabilities
    probe_path = Path(cfg.results_dir) / 'arrays' / f'{dataset}_{model}_K{K}_seed{seed}_pernode.npz'
    
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe results not found: {probe_path}")
    
    return np.load(probe_path)


def load_classifier_outputs(dataset, model, K, seed, loss_type='exponential'):
    """Load classifier head outputs."""
    logits_path = Path(cfg.classifier_heads_dir) / loss_type / dataset / model / f'seed_{seed}' / f'K_{K}' / 'layer_logits.npz'
    probs_path = Path(cfg.classifier_heads_dir) / loss_type / dataset / model / f'seed_{seed}' / f'K_{K}' / 'layer_probs.npz'
    
    if not logits_path.exists():
        raise FileNotFoundError(f"Classifier outputs not found: {logits_path}")
    
    logits_data = np.load(logits_path)
    probs_data = np.load(probs_path)
    
    return logits_data, probs_data


def plot_layer_comparison(ax, probe_probs, classifier_probs, layer_idx, num_classes, split='test'):
    """
    Plot probability distribution comparison for a single layer.
    
    Shows histograms of predicted probabilities for the correct class.
    """
    # Get probabilities for correct class (assuming we have labels)
    # For simplicity, plot distribution of max probabilities
    
    probe_max_probs = probe_probs.max(axis=1)
    classifier_max_probs = classifier_probs.max(axis=1)
    
    # Create side-by-side histograms
    bins = np.linspace(0, 1, 20)
    
    ax.hist(probe_max_probs, bins=bins, alpha=0.6, label='Linear Probe (Frozen)', 
            color='steelblue', edgecolor='black', linewidth=0.5)
    ax.hist(classifier_max_probs, bins=bins, alpha=0.6, label='Classifier Head (Trained)',
            color='coral', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel(f'Max Probability', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'Layer {layer_idx}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add statistics as text
    stats_text = (
        f'Probe: μ={probe_max_probs.mean():.3f}, σ={probe_max_probs.std():.3f}\n'
        f'Classifier: μ={classifier_max_probs.mean():.3f}, σ={classifier_max_probs.std():.3f}'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))


def plot_probes_vs_classifier_heads(dataset, model, K, seed, loss_type='exponential', split='test'):
    """
    Create comprehensive comparison plot for all layers.
    
    Args:
        dataset: Dataset name
        model: Model name
        K: Number of layers
        seed: Random seed
        loss_type: Loss type used for classifier heads
        split: Which split to visualize ('train', 'val', or 'test')
    """
    print(f"\n{'='*60}")
    print(f"Creating comparison plot:")
    print(f"  Dataset: {dataset}, Model: {model}, K={K}, seed={seed}")
    print(f"  Loss type: {loss_type}, Split: {split}")
    print(f"{'='*60}\n")
    
    # Load probe results
    try:
        probe_results = load_probe_results(dataset, model, K, seed)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("  Please run probing first: python -m src.probe --dataset {dataset} --model {model} --K {K} --seed {seed}")
        return
    
    # Load classifier outputs
    try:
        logits_data, probs_data = load_classifier_outputs(dataset, model, K, seed, loss_type)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"  Please extract classifier outputs first:")
        print(f"  python -m src.extract_classifier_outputs --dataset {dataset} --model {model} --K {K} --seed {seed} --loss-type {loss_type}")
        return
    
    num_classes = int(probs_data['num_classes'])
    num_layers = K + 1  # k=0 to K
    
    # Create figure with subplots for each layer
    fig_height = 3 * num_layers
    fig = plt.figure(figsize=(12, fig_height))
    gs = gridspec.GridSpec(num_layers, 1, figure=fig, hspace=0.4)
    
    for k in range(num_layers):
        ax = fig.add_subplot(gs[k, 0])
        
        # Get probe probabilities for this layer from .npz file
        # Keys in the .npz file are like: 'p_val_0', 'p_test_0', etc.
        probe_key = f'p_{split}_{k}'
        if probe_key not in probe_results:
            print(f"  Warning: No probe probabilities found for layer {k} (key: {probe_key}), skipping")
            continue
        
        probe_probs = probe_results[probe_key]
        
        # Get classifier head probabilities for this layer
        classifier_probs = probs_data[f'{split}_probs_{k}']
        
        # Plot comparison
        plot_layer_comparison(ax, probe_probs, classifier_probs, k, num_classes, split)
    
    # Overall title
    fig.suptitle(
        f'{dataset} {model} K={K} seed={seed} - Linear Probe vs Classifier Head ({loss_type} loss)\n'
        f'Split: {split}',
        fontsize=14, fontweight='bold', y=0.995
    )
    
    # Save figure
    output_dir = Path(cfg.figures_dir) / 'probe_vs_classifier' / loss_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f'{dataset}_{model}_k{K}_seed{seed}_comparison.pdf'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description='Compare linear probes vs classifier heads')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--model', type=str, required=True, help='Model name (GCN, GAT, GraphSAGE)')
    parser.add_argument('--K', type=int, default=8, help='Number of layers')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--loss-type', type=str, default='exponential',
                        choices=['exponential', 'class-weighted'],
                        help='Loss type used for classifier heads')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to visualize')
    
    args = parser.parse_args()
    
    plot_probes_vs_classifier_heads(
        dataset=args.dataset,
        model=args.model,
        K=args.K,
        seed=args.seed,
        loss_type=args.loss_type,
        split=args.split
    )


if __name__ == '__main__':
    main()
