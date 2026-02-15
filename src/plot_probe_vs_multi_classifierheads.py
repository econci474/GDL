"""Compare probe results vs multiple classifier head variants (scatter plot)."""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from src.datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description='Compare probe vs classifier heads (scatter)')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--K', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--ch-loss-types', type=str, nargs='+', required=True,
                       help='Classifier head loss types to compare')
    args = parser.parse_args()
    
    # Load dataset for test mask
    data, num_classes, _ = load_dataset(args.dataset, root_dir=cfg.data_dir)
    test_mask = data.test_mask.numpy() if hasattr(data.test_mask, 'numpy') else data.test_mask
    
    # Load probe results
    probe_path = Path(cfg.tables_dir) / f'{args.dataset}_{args.model}_K{args.K}_seed{args.seed}_probe.csv'
    probe_df = pd.read_csv(probe_path)
    
    # Create figure with subplots for each layer
    num_layers = args.K + 1
    fig, axes = plt.subplots(1, num_layers, figsize=(4 * num_layers, 4))
    if num_layers == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(range(len(args.ch_loss_types) + 1))
    
    for k in range(num_layers):
        ax = axes[k]
        
        # Get probe test accuracy for this layer
        probe_acc = probe_df.loc[probe_df['k'] == k, 'test_acc'].values[0]
        
        # Plot each classifier head variant
        for idx, loss_type in enumerate(args.ch_loss_types):
            # Load classifier head test log
            ch_path = Path(cfg.classifier_heads_dir) / loss_type / args.dataset / args.model / f'seed_{args.seed}' / f'K_{args.K}' / 'test_log.csv'
            
            if ch_path.exists():
                ch_df = pd.read_csv(ch_path)
                # Find test accuracy for this layer k
                ch_row = ch_df[ch_df['k'] == k]
                if len(ch_row) > 0:
                    ch_acc = ch_row.iloc[0]['test_acc']
                    
                    # Plot point
                    label_short = loss_type.replace('ce_plus_R_R1.0_', '').replace('_', ' ')
                    ax.scatter(probe_acc, ch_acc, s=100, alpha=0.7, 
                             color=colors[idx], label=label_short, edgecolors='black', linewidths=1)
        
        # Plot diagonal line (y=x)
        max_val = max(probe_acc, max([ax.get_ylim()[1] for ax in axes]))
        min_val = min(probe_acc, min([ax.get_ylim()[0] for ax in axes]))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=1)
        
        # Formatting
        ax.set_xlabel('Probe Test Acc', fontsize=10)
        if k == 0:
            ax.set_ylabel('Classifier Head Test Acc', fontsize=10)
        ax.set_title(f'Layer k={k}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal', adjustable='box')
        
        if k == num_layers - 1:
            ax.legend(loc='lower right', fontsize=8)
    
    plt.suptitle(f'{args.dataset} {args.model} K={args.K} seed={args.seed}: Probe vs Classifier Heads',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_dir = Path(cfg.figures_dir) / 'probe_vs_classifierheads_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loss_str = '_vs_'.join([lt.replace('ce_plus_R_R1.0_', '') for lt in args.ch_loss_types])
    output_path = output_dir / f'{args.dataset}_{args.model}_K{args.K}_seed{args.seed}_{loss_str}_scatter.pdf'
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Scatter plot saved to: {output_path}")


if __name__ == '__main__':
    main()
