"""Generate comparison training curves for K_max vs k* with NEW directory structure."""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

sns.set_style('whitegrid')


def plot_training_comparison(dataset_name, model_name, K_max, k_star, config, seeds=None):
    """
    Create 2-row comparison: top row=K_max, bottom row=k*.
    Uses NEW directory structure: dataset/model/seed_X/K_Y/
    """
    if seeds is None:
        seeds = config['seeds']
    
    n_seeds = len(seeds)
    fig, axes = plt.subplots(2, n_seeds, figsize=(6*n_seeds, 10))
    
    if n_seeds == 1:
        axes = axes.reshape(2, 1)
    
    for seed_idx, seed in enumerate(seeds):
        # TOP ROW: K_max
        log_kmax = Path(config['runs_dir']) / dataset_name / model_name / f'seed_{seed}' / f'K_{K_max}' / 'train_log.csv'
        
        if log_kmax.exists():
            df_kmax = pd.read_csv(log_kmax)
            ax = axes[0, seed_idx]
            
            ax.plot(df_kmax['epoch'], df_kmax['train_loss'], 'o-',
                   label='Train Loss', linewidth=2, markersize=3, alpha=0.7, color='#2E86AB')
            ax.plot( df_kmax['epoch'], df_kmax['val_loss'], 's-',
                   label='Val Loss', linewidth=2, markersize=3, alpha=0.7, color='#F18F01')
            
            best_epoch = df_kmax['val_loss'].idxmin()
            test_acc = df_kmax.loc[best_epoch, 'test_acc']
            val_loss = df_kmax.loc[best_epoch, 'val_loss']
            
            ax.axvline(best_epoch + 1, color='red', linestyle='--', linewidth=2, alpha=0.5)
            
            info_text = (
                f"K_max = {K_max}\n"
                f"Best Epoch: {best_epoch+1}\n"
                f"Test Acc: {test_acc:.3f}\n"
                f"Val Loss: {val_loss:.3f}"
            )
            
            ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4))
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss (NLL)', fontsize=12)
            ax.set_title(f'K_max={K_max}, Seed {seed}', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # BOTTOM ROW: k*
        log_kstar = Path(config['runs_dir']) / dataset_name / model_name / f'seed_{seed}' / f'K_{k_star}' / 'train_log.csv'
        
        if log_kstar.exists():
            df_kstar = pd.read_csv(log_kstar)
            ax = axes[1, seed_idx]
            
            ax.plot(df_kstar['epoch'], df_kstar['train_loss'], 'o-',
                   label='Train Loss', linewidth=2, markersize=3, alpha=0.7, color='#2E86AB')
            ax.plot(df_kstar['epoch'], df_kstar['val_loss'], 's-',
                   label='Val Loss', linewidth=2, markersize=3, alpha=0.7, color='#F18F01')
            
            best_epoch = df_kstar['val_loss'].idxmin()
            test_acc = df_kstar.loc[best_epoch, 'test_acc']
            val_loss = df_kstar.loc[best_epoch, 'val_loss']
            
            ax.axvline(best_epoch + 1, color='red', linestyle='--', linewidth=2, alpha=0.5)
            
            info_text = (
                f"k* = {k_star}\n"
                f"Best Epoch: {best_epoch+1}\n"
                f"Test Acc: {test_acc:.3f}\n"
                f"Val Loss: {val_loss:.3f}"
            )
            
            ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='bottom', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.4))
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss (NLL)', fontsize=12)
            ax.set_title(f'k*={k_star} (Optimized), Seed {seed}', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Training Comparison: K_max={K_max} vs k*={k_star}\n{dataset_name} - {model_name}',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save with NEW structure: dataset/model/K_max/
    output_dir = Path(config['figures_dir']) / dataset_name / model_name / f'K_{K_max}'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{dataset_name}_{model_name}_k{K_max}_training_comparison_vs_k{k_star}.png'
    
    fig.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"  ✓ Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--K_max', type=int, required=True)
    parser.add_argument('--k_star', type=int, required=True)
    
    args = parser.parse_args()
    
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    print(f"\n{'='*70}")
    print(f"Training Comparison: K_max={args.K_max} vs k*={args.k_star}")
    print(f"{args.dataset} - {args.model}")
    print(f"{'='*70}\n")
    
    plot_training_comparison(args.dataset, args.model, args.K_max, args.k_star, config)
    
    print(f"\n✅ Comparison complete!\n")


if __name__ == '__main__':
    main()
