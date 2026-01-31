"""Generate plots for a specific K value showing accuracy and NLL vs depth."""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

sns.set_style('whitegrid')


def plot_metrics_vs_depth(dataset_name, model_name, K, config, seeds=None):
    """
    Generate accuracy and NLL vs depth plots for a specific K value.
    Outputs to: results/figures/dataset/model/K_X/
    """
    if seeds is None:
        seeds = config['seeds']
    
    # Load probe results for all seeds
    all_val_acc = []
    all_test_acc = []
    all_val_nll = []
    
    for seed in seeds:
        probe_file = Path(config['tables_dir']) / f'{dataset_name}_{model_name}_K{K}_seed{seed}_probe.csv'
        if not probe_file.exists():
            print(f"  ⚠ Probe file not found: {probe_file}")
            continue
        
        df = pd.read_csv(probe_file)
        all_val_acc.append(df['val_acc'].values)
        all_test_acc.append(df['test_acc'].values)
        all_val_nll.append(df['val_nll'].values)
    
    if len(all_val_acc) == 0:
        print(f"  ⚠ No probe data found for K={K}")
        return
    
    depths = df['k'].values
    
    # Calculate means and stds
    val_acc_mean = np.mean(all_val_acc, axis=0)
    val_acc_std = np.std(all_val_acc, axis=0)
    test_acc_mean = np.mean(all_test_acc, axis=0)
    test_acc_std = np.std(all_test_acc, axis=0)
    val_nll_mean = np.mean(all_val_nll, axis=0)
    val_nll_std = np.std(all_val_nll, axis=0)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy vs Depth
    ax1.plot(depths, val_acc_mean, 'o-', label='Validation',
            linewidth=2, markersize=6, color='#2E86AB')
    ax1.fill_between(depths, val_acc_mean - val_acc_std, val_acc_mean + val_acc_std,
                     alpha=0.2, color='#2E86AB')
    
    ax1.plot(depths, test_acc_mean, 's-', label='Test',
            linewidth=2, markersize=6, color='#F18F01')
    ax1.fill_between(depths, test_acc_mean - test_acc_std, test_acc_mean + test_acc_std,
                     alpha=0.2, color='#F18F01')
    
    ax1.set_xlabel('Depth (k)', fontsize=13)
    ax1.set_ylabel('Accuracy', fontsize=13)
    ax1.set_title(f'Accuracy vs Depth (K={K}, ±1 std, n_seeds={len(seeds)})',
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    ax1.set_xticks(depths)
    
    # Plot 2: Validation NLL vs Depth
    ax2.plot(depths, val_nll_mean, 'o-', linewidth=2, markersize=6, color='#A23B72')
    ax2.fill_between(depths, val_nll_mean - val_nll_std, val_nll_mean + val_nll_std,
                     alpha=0.2, color='#A23B72')
    
    # Mark k* (minimum NLL)
    k_star_idx = np.argmin(val_nll_mean)
    k_star = depths[k_star_idx]
    ax2.axvline(k_star, color='orange', linestyle='--', linewidth=2,
               label=f'k*={k_star}', alpha=0.8)
    
    ax2.set_xlabel('Depth (k)', fontsize=13)
    ax2.set_ylabel('Validation NLL', fontsize=13)
    ax2.set_title(f'Validation NLL vs Depth (K={K}, ±1 std, n_seeds={len(seeds)})',
                 fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(depths)
    
    fig.suptitle(f'{dataset_name} - {model_name} (K_max={K})',
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save to dataset/model/K_X/ directory
    output_dir = Path(config['figures_dir']) / dataset_name / model_name / f'K_{K}'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'accuracy_nll_vs_depth.pdf'
    
    fig.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"  ✓ Saved: {output_file}")
    
    # Print summary
    print(f"\n  Summary for K={K}:")
    print(f"    k* (min val_nll) = {k_star}")
    print(f"    Test accuracy at k*: {test_acc_mean[k_star_idx]:.3f} ± {test_acc_std[k_star_idx]:.3f}")
    print(f"    Test accuracy at k={K}: {test_acc_mean[-1]:.3f} ± {test_acc_std[-1]:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Generate accuracy and NLL plots for specific K')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--K', type=int, required=True,
                       help='K value to plot')
    
    args = parser.parse_args()
    
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    print(f"\n{'='*70}")
    print(f"Generating Plots for K={args.K}: {args.dataset} - {args.model}")
    print(f"{'='*70}\n")
    
    plot_metrics_vs_depth(args.dataset, args.model, args.K, config)
    
    print(f"\n✅ Plots complete!\n")


if __name__ == '__main__':
    main()
