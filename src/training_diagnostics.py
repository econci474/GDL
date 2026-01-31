"""Training diagnostic plots: NLL/loss curves by epoch."""

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


def plot_training_curves(dataset_name, model_name, K, config, seeds=None):
    """
    Plot training and validation loss curves for each seed.
    Creates a multi-panel figure with one panel per seed.
    """
    if seeds is None:
        seeds = config['seeds']
    
    n_seeds = len(seeds)
    fig, axes = plt.subplots(1, n_seeds, figsize=(6*n_seeds, 5))
    
    if n_seeds == 1:
        axes = [axes]
    
    for idx, seed in enumerate(seeds):
        log_file = Path(config['runs_dir']) / dataset_name / model_name / f'seed_{seed}' / 'train_log.csv'
        
        if not log_file.exists():
            print(f"  ⚠ Training log not found for seed {seed}")
            continue
        
        df = pd.read_csv(log_file)
        
        ax = axes[idx]
        
        # Plot losses
        ax.plot(df['epoch'], df['train_loss'], 'o-', label='Train Loss',
                linewidth=2, markersize=4, alpha=0.7, color='#2E86AB')
        ax.plot(df['epoch'], df['val_loss'], 's-', label='Val Loss',
                linewidth=2, markersize=4, alpha=0.7, color='#F18F01')
        
        # Mark best epoch
        best_epoch = df['val_loss'].idxmin()
        best_val_loss = df.loc[best_epoch, 'val_loss']
        best_train_loss = df.loc[best_epoch, 'train_loss']
        
        ax.axvline(best_epoch + 1, color='red', linestyle='--', linewidth=2,
                   alpha=0.5, label=f'Best Epoch={best_epoch+1}')
        
        # Add info text
        last_epoch = len(df)
        lr = config['lr']
        patience = config['patience']
        
        info_text = (
            f"Seed: {seed}\n"
            f"LR: {lr}\n"
            f"Patience: {patience}\n"
            f"Best Epoch: {best_epoch+1}/{last_epoch}\n"
            f"Best Val Loss: {best_val_loss:.3f}\n"
            f"Train Loss @ Best: {best_train_loss:.3f}"
        )
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (NLL)', fontsize=12)
        ax.set_title(f'Seed {seed}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Training Curves: {dataset_name} - {model_name} (K={K})',
                 fontsize=15, fontweight='bold', y=1.02)
    
    output_dir = Path(config['figures_dir']) / dataset_name / model_name / f'K_{K}'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{dataset_name}_{model_name}_k{K}_training_curves.pdf'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"  ✓ Saved: {output_file}")


def plot_accuracy_curves(dataset_name, model_name, K, config, seeds=None):
    """
    Plot training, validation, and test accuracy curves for each seed.
    """
    if seeds is None:
        seeds = config['seeds']
    
    n_seeds = len(seeds)
    fig, axes = plt.subplots(1, n_seeds, figsize=(6*n_seeds, 5))
    
    if n_seeds == 1:
        axes = [axes]
    
    for idx, seed in enumerate(seeds):
        log_file = Path(config['runs_dir']) / dataset_name / model_name / f'seed_{seed}' / f'K_{K}' / 'train_log.csv'
        
        if not log_file.exists():
            print(f"  ⚠ Training log not found for seed {seed}")
            continue
        
        df = pd.read_csv(log_file)
        
        ax = axes[idx]
        
        # Plot accuracies
        ax.plot(df['epoch'], df['train_acc'], 'o-', label='Train Acc',
                linewidth=2, markersize=3, alpha=0.7, color='#2E86AB')
        ax.plot(df['epoch'], df['val_acc'], 's-', label='Val Acc',
                linewidth=2, markersize=3, alpha=0.7, color='#F18F01')
        ax.plot(df['epoch'], df['test_acc'], '^-', label='Test Acc',
                linewidth=2, markersize=3, alpha=0.7, color='#A23B72')
        
        # Mark best epoch (by val_loss)
        best_epoch = df['val_loss'].idxmin()
        ax.axvline(best_epoch + 1, color='red', linestyle='--', linewidth=2,
                   alpha=0.5, label=f'Best Epoch={best_epoch+1}')
        
        # Show accuracies at best epoch
        test_acc_at_best = df.loc[best_epoch, 'test_acc']
        val_acc_at_best = df.loc[best_epoch, 'val_acc']
        
        # Show final test accuracy
        final_test_acc = df['test_acc'].iloc[-1]
        
        info_text = (
            f"Seed: {seed}\n"
            f"Test Acc @ Best: {test_acc_at_best:.3f}\n"
            f"Val Acc @ Best: {val_acc_at_best:.3f}\n"
            f"Final Test Acc: {final_test_acc:.3f}"
        )
        
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'Seed {seed}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    fig.suptitle(f'Accuracy Curves: {dataset_name} - {model_name} (K={K})',
                 fontsize=15, fontweight='bold', y=0.995)
    
    output_dir = Path(config['figures_dir']) / dataset_name / model_name / f'K_{K}'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{dataset_name}_{model_name}_k{K}_accuracy_curves.pdf'
    fig.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"  ✓ Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate training diagnostic plots')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--K', type=int, required=True,
                       help='K value (model depth)')
    
    args = parser.parse_args()
    
    # Convert config module to dict
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    print(f"\n{'='*70}")
    print(f"Generating Training Diagnostic Plots: {args.dataset} {args.model} (K={args.K})")
    print(f"{'='*70}\n")
    
    print("  Generating training loss curves...")
    plot_training_curves(args.dataset, args.model, args.K, config)
    
    print("  Generating accuracy curves...")
    plot_accuracy_curves(args.dataset, args.model, args.K, config)
    
    print(f"\n{'='*60}")
    print("✓ Training diagnostic plots complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
