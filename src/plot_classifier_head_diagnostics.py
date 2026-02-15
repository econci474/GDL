"""
Generate training diagnostic plots for classifier head models.
Adapted version that reads from classifier_heads directories.
"""

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


def has_splits(dataset_name, model_name, K, seed, loss_type, config):
    """
    Check if this dataset/model/K/seed combination uses split-based structure.
    Returns True if split_0 subdirectory exists, False otherwise.
    """
    heads_dir = Path(config['classifier_heads_dir']) / loss_type / dataset_name / model_name / f'seed_{seed}' / f'K_{K}'
    split_0_dir = heads_dir / 'split_0'
    return split_0_dir.exists()


def plot_per_split_diagnostics(dataset_name, model_name, K, loss_type, config, seeds=None, num_splits=10):
    """
    Plot combined training diagnostics for classifier heads with per-split structure.
    Creates one large figure with 2 rows (accuracy, loss) and num_splits columns.
    """
    if seeds is None:
        seeds = config['seeds']
    
    for seed in seeds:
        fig, axes = plt.subplots(2, num_splits, figsize=(3*num_splits, 8))
        
        for split_idx in range(num_splits):
            log_file = Path(config['classifier_heads_dir']) / loss_type / dataset_name / model_name / f'seed_{seed}' / f'K_{K}' / f'split_{split_idx}' / 'train_log.csv'
            
            if not log_file.exists():
                print(f"  ⚠ Training log not found for seed {seed}, split {split_idx}")
                continue
            
            df = pd.read_csv(log_file)
            best_epoch = df['val_loss'].idxmin()
            
            # Top row: Accuracy curves
            ax = axes[0, split_idx]
            ax.plot(df['epoch'], df['train_acc'], 'o-', label='Train',
                    linewidth=1.5, markersize=2, alpha=0.7, color='#2E86AB')
            ax.plot(df['epoch'], df['val_acc'], 's-', label='Val',
                    linewidth=1.5, markersize=2, alpha=0.7, color='#F18F01')
            ax.axvline(best_epoch + 1, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
            
            train_acc_at_best = df.loc[best_epoch, 'train_acc']
            val_acc_at_best = df.loc[best_epoch, 'val_acc']
            
            ax.text(0.5, 0.02, f'Val: {val_acc_at_best:.2f}', 
                    transform=ax.transAxes, fontsize=8, ha='center',
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            
            ax.set_xlabel('Epoch', fontsize=9)
            if split_idx == 0:
                ax.set_ylabel('Accuracy', fontsize=10)
            ax.set_title(f'Split {split_idx}', fontsize=10, fontweight='bold')
            if split_idx == 0:
                ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
            
            # Bottom row: Loss curves
            ax = axes[1, split_idx]
            ax.plot(df['epoch'], df['train_loss'], 'o-', label='Train',
                    linewidth=1.5, markersize=2, alpha=0.7, color='#2E86AB')
            ax.plot(df['epoch'], df['val_loss'], 's-', label='Val',
                    linewidth=1.5, markersize=2, alpha=0.7, color='#F18F01')
            ax.axvline(best_epoch + 1, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
            
            best_val_loss = df.loc[best_epoch, 'val_loss']
            last_epoch = len(df)
            
            # Add hyperparameters to first split only
            if split_idx == 0:
                # Try to read hyperparameters from CSV (saved in newer runs)
                # Fall back to config for backward compatibility with older runs
                if 'lr' in df.columns:
                    lr = df.loc[0, 'lr']  # Same for all epochs
                    patience = df.loc[0, 'patience']
                    max_epochs = df.loc[0, 'max_epochs']
                else:
                    lr = config.get('classifier_lr', config.get('lr', 0.01))
                    patience = config.get('classifier_patience', config.get('patience', 100))
                    max_epochs = config.get('max_epochs', 200)
                
                info_text = (
                    f'LR: {lr}\\n'
                    f'Patience: {patience}\\n'
                    f'Max Epochs: {max_epochs}\\n'
                    f'Best: {best_epoch+1}/{last_epoch}\\n'
                    f'NLL: {best_val_loss:.2f}'
                )
            else:
                info_text = f'Best: {best_epoch+1}\\nNLL: {best_val_loss:.2f}'
            
            ax.text(0.5, 0.98, info_text,
                    transform=ax.transAxes, fontsize=8, ha='center',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            ax.set_xlabel('Epoch', fontsize=9)
            if split_idx == 0:
                ax.set_ylabel('Loss (NLL)', fontsize=10)
            if split_idx == 0:
                ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'Per-Split Classifier Head Diagnostics ({loss_type}): {dataset_name} - {model_name} (K={K}, Seed={seed})',
                     fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        output_dir = Path(config['figures_dir']) / 'training_diagnostics' / loss_type / dataset_name / model_name / f'K_{K}'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'{dataset_name}_{model_name}_k{K}_seed{seed}_per_split_diagnostics.png'
        fig.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        print(f"  ✓ Saved: {output_file}")


def plot_classifier_head_diagnostics(dataset_name, model_name, K, loss_type, config, seeds=None):
    """
    Plot training diagnostics for classifier head models.
    Creates one 2-row figure: accuracy (top), loss (bottom).
    
    Args:
        dataset_name: Dataset name (Cora, PubMed, etc.)
        model_name: Model name (GCN, GAT, etc.)
        K: Number of layers
        loss_type: Loss type directory name
        config: Configuration dict
        seeds: List of seeds to plot (None = all)
    """
    if seeds is None:
        seeds = config['seeds']
    
    n_seeds = len(seeds)
    
    # Create combined figure: 2 rows x n_seeds columns
    fig, axes = plt.subplots(2, n_seeds, figsize=(6*n_seeds, 10))
    
    if n_seeds == 1:
        axes = axes.reshape(2, 1)
    
    for idx, seed in enumerate(seeds):
        log_file = Path(cfg.classifier_heads_dir) / loss_type / dataset_name / model_name / f'seed_{seed}' / f'K_{K}' / 'train_log.csv'
        
        if not log_file.exists():
            print(f"  ⚠ Training log not found for seed {seed}: {log_file}")
            continue
        
        df = pd.read_csv(log_file)
        best_epoch = df['val_loss'].idxmin()
        
        # === TOP ROW: ACCURACY ===
        ax = axes[0, idx]
        
        ax.plot(df['epoch'], df['train_acc'], 'o-', label='Train Acc',
                linewidth=2, markersize=3, alpha=0.7, color='#2E86AB')
        ax.plot(df['epoch'], df['val_acc'], 's-', label='Val Acc',
                linewidth=2, markersize=3, alpha=0.7, color='#F18F01')
        ax.axvline(best_epoch + 1, color='red', linestyle='--', linewidth=2,
                   alpha=0.5, label=f'Best Epoch={best_epoch+1}')
        
        train_acc_at_best = df.loc[best_epoch, 'train_acc']
        val_acc_at_best = df.loc[best_epoch, 'val_acc']
        
        info_text = (
            f"Seed: {seed}\\n"
            f"Train Acc @ Best: {train_acc_at_best:.3f}\\n"
            f"Val Acc @ Best: {val_acc_at_best:.3f}"
        )
        
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(f'Seed {seed}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # === BOTTOM ROW: LOSS ===
        ax = axes[1, idx]
        
        ax.plot(df['epoch'], df['train_loss'], 'o-', label='Train Loss',
                linewidth=2, markersize=4, alpha=0.7, color='#2E86AB')
        ax.plot(df['epoch'], df['val_loss'], 's-', label='Val Loss',
                linewidth=2, markersize=4, alpha=0.7, color='#F18F01')
        ax.axvline(best_epoch + 1, color='red', linestyle='--', linewidth=2,
                   alpha=0.5)
        
        best_val_loss = df.loc[best_epoch, 'val_loss']
        best_train_loss = df.loc[best_epoch, 'train_loss']
        last_epoch = len(df)
        
        # Try to read hyperparameters from CSV (saved in newer runs)
        # Fall back to config for backward compatibility with older runs
        if 'lr' in df.columns:
            lr = df.loc[0, 'lr']  # Same for all epochs
            patience = df.loc[0, 'patience']
            max_epochs = df.loc[0, 'max_epochs']
        else:
            lr = config.get('classifier_lr', config.get('lr', 0.01))
            patience = config.get('classifier_patience', config.get('patience', 100))
            max_epochs = config.get('max_epochs', 200)
        
        info_text = (
            f"LR: {lr}\\n"
            f"Patience: {patience}\\n"
            f"Max Epochs: {max_epochs}\\n"
            f"Best Epoch: {best_epoch+1}/{last_epoch}\\n"
            f"Best Val Loss: {best_val_loss:.3f}\\n"
            f"Train Loss @ Best: {best_train_loss:.3f}"
        )
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss (NLL)', fontsize=11)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Training Diagnostics ({loss_type}): {dataset_name} - {model_name} (K={K})',
                 fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    output_dir = Path(config['figures_dir']) / 'training_diagnostics' / loss_type / dataset_name / model_name / f'K_{K}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f'{dataset_name}_{model_name}_k{K}_training_diagnostics.png'
    fig.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate classifier head training diagnostic plots')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--K', type=str, default='all', help='K value or "all"')
    parser.add_argument('--loss-type', type=str, default='class-weighted',
                       help='Loss type directory name (e.g., exponential, class-weighted, ce_plus_R_R1.0_hard)')
    
    args = parser.parse_args()
    
    # Convert config module to dict
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    # Determine K values
    if args.K.lower() == 'all':
        K_values = list(range(9))  # 0-8
    else:
        K_values = [int(args.K)]
    
    # Determine loss types
    if args.loss_type == 'all':
        loss_types = ['exponential', 'class-weighted']
    else:
        loss_types = [args.loss_type]
    
    print(f"\n{'='*70}")
    print(f"Generating Classifier Head Training Diagnostics: {args.dataset}")
    print(f"K values: {K_values}")
    print(f"Loss types: {loss_types}")
    print(f"{'='*70}\n")
    
    for loss_type in loss_types:
        for K in K_values:
            print(f"\nProcessing {loss_type} K={K}...")
            
            # Detect if dataset uses split-based structure
            first_seed = config['seeds'][0] if config['seeds'] else 0
            uses_splits = has_splits(args.dataset, args.model, K, first_seed, loss_type, config)
            
            if uses_splits:
                print(f"  Detected per-split structure. Generating diagnostics for splits...")
                plot_per_split_diagnostics(args.dataset, args.model, K, loss_type, config)
            else:
                plot_classifier_head_diagnostics(args.dataset, args.model, K, loss_type, config)
    
    print(f"\n{'='*60}")
    print("✓ Training diagnostic plots complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
