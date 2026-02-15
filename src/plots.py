"""Enhanced visualization of entropy, accuracy, NLL, and probability dynamics vs depth."""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

from src.datasets import load_dataset
from src.metrics import entropy_from_probs

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def plot_accuracy_vs_depth(dataset_name, model_name, K, config, seeds=None):
    """Plot test AND validation accuracy vs depth with ±1 std."""
    if seeds is None:
        seeds = config['seeds']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_val_acc = []
    all_test_acc = []
    
    for seed in seeds:
        probe_file = Path(config['tables_dir']) / f'{dataset_name}_{model_name}_K{K}_seed{seed}_probe.csv'
        if not probe_file.exists():
            continue
        
        df = pd.read_csv(probe_file)
        all_val_acc.append(df['val_acc'].values)
        all_test_acc.append(df['test_acc'].values)
    
    if len(all_val_acc) == 0:
        print(f"⚠ No data found for {dataset_name} {model_name}")
        return
    
    depths = df['k'].values
    val_mean = np.mean(all_val_acc, axis=0)
    val_std = np.std(all_val_acc, axis=0)
    test_mean = np.mean(all_test_acc, axis=0)
    test_std = np.std(all_test_acc, axis=0)
    
    # Plot with shaded regions
    ax.plot(depths, val_mean, 'o-', label='Validation', linewidth=2, markersize=6, color='#2E86AB')
    ax.fill_between(depths, val_mean - val_std, val_mean + val_std, alpha=0.2, color='#2E86AB')
    
    ax.plot(depths, test_mean, 's-', label='Test', linewidth=2, markersize=6, color='#F18F01')
    ax.fill_between(depths, test_mean - test_std, test_mean + test_std, alpha=0.2, color='#F18F01')
    
    ax.set_xlabel('Depth (k)', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_title(f'Accuracy vs Depth (±1 std, n_seeds={len(seeds)})\n{dataset_name} - {model_name} (K={K})', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    output_dir = Path(config['figures_dir']) / dataset_name / model_name / f'K_{K}'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{dataset_name}_{model_name}_k{K}_accuracy_vs_depth.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Saved: {output_file}")


def plot_entropy_vs_depth(dataset_name, model_name, K, config, seeds=None):
    """Plot mean entropy vs depth with node count specified."""
    if seeds is None:
        seeds = config['seeds']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_val_entropy = []
    all_test_entropy = []
    
    # Get dataset info for node counts
    data, _, _ = load_dataset(dataset_name)
    n_val_nodes = data.val_mask.sum().item()
    n_test_nodes = data.test_mask.sum().item()
    
    for seed in seeds:
        probe_file = Path(config['tables_dir']) / f'{dataset_name}_{model_name}_K{K}_seed{seed}_probe.csv'
        if not probe_file.exists():
            continue
        
        df = pd.read_csv(probe_file)
        all_val_entropy.append(df['val_entropy_mean'].values)
        all_test_entropy.append(df['test_entropy_mean'].values)
    
    if len(all_val_entropy) == 0:
        print(f"No data found for {dataset_name} {model_name}")
        return
    
    depths = df['k'].values
    val_mean = np.mean(all_val_entropy, axis=0)
    val_std = np.std(all_val_entropy, axis=0)
    test_mean = np.mean(all_test_entropy, axis=0)
    test_std = np.std(all_test_entropy, axis=0)
    
    # Plot with shaded regions
    ax.plot(depths, val_mean, 'o-', label=f'Validation (n={n_val_nodes})', linewidth=2, markersize=6)
    ax.fill_between(depths, val_mean - val_std, val_mean + val_std, alpha=0.2)
    
    ax.plot(depths, test_mean, 's-', label=f'Test (n={n_test_nodes})', linewidth=2, markersize=6)
    ax.fill_between(depths, test_mean - test_std, test_mean + test_std, alpha=0.2)
    
    ax.set_xlabel('Depth (k)', fontsize=13)
    ax.set_ylabel('Mean Predictive Entropy (nats)', fontsize=13)
    ax.set_title(f'Predictive Entropy vs Depth (±1 std, n_seeds={len(seeds)})\n{dataset_name} - {model_name} (K={K})', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    output_dir = Path(config['figures_dir']) / dataset_name / model_name / f'K_{K}'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{dataset_name}_{model_name}_k{K}_entropy_vs_depth.png'
    fig.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Saved: {output_file}")


def plot_nll_vs_depth(dataset_name, model_name, K, config, seeds=None):
    """Plot validation NLL vs depth with ±std and mark selected k*."""
    if seeds is None:
        seeds = config['seeds']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_val_nll = []
    all_k_star = []
    
    for seed in seeds:
        probe_file = Path(config['tables_dir']) / f'{dataset_name}_{model_name}_K{K}_seed{seed}_probe.csv'
        if not probe_file.exists():
            continue
        
        df = pd.read_csv(probe_file)
        all_val_nll.append(df['val_nll'].values)
        
        # Find k*
        k_star = df.loc[df['val_nll'].idxmin(), 'k']
        all_k_star.append(k_star)
    
    if len(all_val_nll) == 0:
        print(f"No data found for {dataset_name} {model_name}")
        return
    
    depths = df['k'].values
    nll_mean = np.mean(all_val_nll, axis=0)
    nll_std = np.std(all_val_nll, axis=0)
    
    ax.plot(depths, nll_mean, 'o-', linewidth=2, markersize=6, color='#A23B72')
    ax.fill_between(depths, nll_mean - nll_std, nll_mean + nll_std, alpha=0.2, color='#A23B72')
    
    # Mark mean k*
    k_star_mean = np.mean(all_k_star)
    ax.axvline(k_star_mean, color='orange', linestyle='--', linewidth=2, 
               label=f'k* = {k_star_mean:.1f}', alpha=0.8)
    
    ax.set_xlabel('Depth (k)', fontsize=13)
    ax.set_ylabel('Validation NLL', fontsize=13)
    ax.set_title(f'Validation NLL vs Depth (±1 std, n_seeds={len(seeds)})\n{dataset_name} - {model_name} (K={K})', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    output_dir = Path(config['figures_dir']) / dataset_name / model_name / f'K_{K}'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{dataset_name}_{model_name}_k{K}_nll_vs_depth.png'
    fig.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Saved: {output_file}")


def plot_correct_incorrect_entropy(dataset_name, model_name, K, config, seeds=None):
    """
    Plot entropy vs depth for correctly vs incorrectly predicted nodes.
    Shows the number of nodes in each category in the legend.
    """
    if seeds is None:
        seeds = config['seeds']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data, _, _ = load_dataset(dataset_name)
    test_mask = data.test_mask.numpy()
    y_test = data.y[test_mask].numpy()
    n_test = test_mask.sum()
    
    all_correct_entropy = []
    all_incorrect_entropy = []
    all_n_correct = []
    all_n_incorrect = []
    
    for seed in seeds:
        probe_file = Path(config['tables_dir']) / f'{dataset_name}_{model_name}_K{K}_seed{seed}_probe.csv'
        if not probe_file.exists():
            continue
        
        df = pd.read_csv(probe_file)
        correct_entropies = df['correct_entropy_mean'].values
        incorrect_entropies = df['incorrect_entropy_mean'].values
        
        all_correct_entropy.append(correct_entropies)
        all_incorrect_entropy.append(incorrect_entropies)
        
        # Calculate number of correct/incorrect at k* (minimum val_nll)
        k_star_idx = df['val_nll'].idxmin()
        test_acc_at_kstar = df.loc[k_star_idx, 'test_acc']
        n_correct = int(test_acc_at_kstar * n_test)
        n_incorrect = n_test - n_correct
        all_n_correct.append(n_correct)
        all_n_incorrect.append(n_incorrect)
    
    if len(all_correct_entropy) == 0:
        print(f"No data found for {dataset_name} {model_name}")
        return
    
    depths = df['k'].values
    correct_mean = np.mean(all_correct_entropy, axis=0)
    correct_std = np.std(all_correct_entropy, axis=0)
    incorrect_mean = np.mean(all_incorrect_entropy, axis=0)
    incorrect_std = np.std(all_incorrect_entropy, axis=0)
    
    # Average node counts at k*
    avg_n_correct = int(np.mean(all_n_correct))
    avg_n_incorrect = int(np.mean(all_n_incorrect))
    
    ax.plot(depths, correct_mean, 'o-', label=f'Correct Predictions (n≈{avg_n_correct})', 
            linewidth=2, markersize=6, color='green')
    ax.fill_between(depths, correct_mean - correct_std, correct_mean + correct_std, 
                     alpha=0.2, color='green')
    
    ax.plot(depths, incorrect_mean, 's-', label=f'Incorrect Predictions (n≈{avg_n_incorrect})', 
            linewidth=2, markersize=6, color='red')
    ax.fill_between(depths, incorrect_mean - incorrect_std, incorrect_mean + incorrect_std, 
                     alpha=0.2, color='red')
    
    ax.set_xlabel('Depth (k)', fontsize=13)
    ax.set_ylabel('Mean Entropy (nats)', fontsize=13)
    ax.set_title(f'Entropy: Correct vs Incorrect (±1 std, n_seeds={len(seeds)}, test_nodes={n_test})\n{dataset_name} - {model_name} (K={K})', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    output_dir = Path(config['figures_dir']) / dataset_name / model_name / f'K_{K}'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{dataset_name}_{model_name}_k{K}_correct_incorrect_entropy_vs_depth.png'
    fig.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Saved: {output_file}")


def plot_depth_selection_comparison(dataset_name, model_name, config):
    """Bar plot comparing K_max vs k* with explicit values."""
    summary_file = Path(config['tables_dir']) / f'summary_{dataset_name}_{model_name}.csv'
    
    if not summary_file.exists():
        print(f"Summary file not found: {summary_file}")
        return
    
    summary_df = pd.read_csv(summary_file)
    
    # Get K_max and k*
    K_max = config['K_max']
    k_star = summary_df['k_star_mean'].values[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Test accuracy comparison
    categories = [f'K_max={K_max}', f'k*={k_star:.1f}']
    acc_means = [
        summary_df['K_max_test_acc_mean'].values[0],
        summary_df['k_star_test_acc_mean'].values[0]
    ]
    acc_stds = [
        summary_df['K_max_test_acc_std'].values[0],
        summary_df['k_star_test_acc_std'].values[0]
    ]
    
    bars1 = ax1.bar(categories, acc_means, yerr=acc_stds, capsize=10, 
                    color=['#2E86AB', '#F18F01'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Test Accuracy', fontsize=13)
    ax1.set_title('Test Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars1, acc_means, acc_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=11)
    
    # Test entropy comparison
    entropy_means = [
        summary_df['K_max_test_entropy_mean'].values[0],
        summary_df['k_star_test_entropy_mean'].values[0]
    ]
    entropy_stds = [
        summary_df['K_max_test_entropy_std'].values[0],
        summary_df['k_star_test_entropy_std'].values[0]
    ]
    
    bars2 = ax2.bar(categories, entropy_means, yerr=entropy_stds, capsize=10,
                    color=['#2E86AB', '#F18F01'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Test Entropy (nats)', fontsize=13)
    ax2.set_title('Test Entropy Comparison', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean, std in zip(bars2, entropy_means, entropy_stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.05,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=11)
    
    fig.suptitle(f'Depth Selection Comparison: {dataset_name} - {model_name}', 
                 fontsize=15, fontweight='bold', y=1.02)
    
    output_file = Path(config['figures_dir']) / f'{dataset_name}_{model_name}_depth_selection_comparison.png'
    fig.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate enhanced plots for entropy analysis')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name')
    parser.add_argument('--K', type=int, required=True,
                       help='K value (model depth)')
    parser.add_argument('--plots', type=str, nargs='+', default=['all'],
                       choices=['entropy', 'accuracy', 'nll', 'correct_incorrect', 
                               'comparison', 'all'],
                       help='Which plots to generate')
    parser.add_argument('--seed', type=int, default=0,
                       help='Seed for single-seed plots (e.g., correct/incorrect)')
    
    args = parser.parse_args()
    
    # Convert config module to dict
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    # Create output directory
    Path(config['figures_dir']).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating Enhanced Plots: {args.dataset} {args.model}")
    print(f"{'='*60}\n")
    
    plot_list = args.plots
    if 'all' in plot_list:
        plot_list = ['entropy', 'accuracy', 'nll', 'correct_incorrect', 'comparison']
    
    if 'entropy' in plot_list:
        print("  Generating entropy vs depth plot...")
        plot_entropy_vs_depth(args.dataset, args.model, args.K, config)
    
    if 'accuracy' in plot_list:
        print("  Generating accuracy vs depth plot...")
        plot_accuracy_vs_depth(args.dataset, args.model, args.K, config)
    
    if 'nll' in plot_list:
        print("  Generating NLL vs depth plot...")
        plot_nll_vs_depth(args.dataset, args.model, args.K, config)
    
    if 'correct_incorrect' in plot_list:
        print("  Generating correct/incorrect entropy plot...")
        plot_correct_incorrect_entropy(args.dataset, args.model, args.K, config)
    
    if 'comparison' in plot_list:
        print("  Generating depth selection comparison plot...")
        plot_depth_selection_comparison(args.dataset, args.model, config)
    
    print(f"\n{'='*60}")
    print(f"All plots generated!")
    print(f"  Output directory: {config['figures_dir']}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
