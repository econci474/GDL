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
        log_file = Path(config['runs_dir']) / dataset_name / model_name / f'seed_{seed}' / f'K_{K}' / 'train_log.csv'
        
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
        
        # Try to read hyperparameters from CSV (saved in newer runs)
        # Fall back to config for backward compatibility with older runs
        if 'lr' in df.columns:
            lr = df.loc[0, 'lr']  # Same for all epochs
            patience = df.loc[0, 'patience']
            max_epochs = df.loc[0, 'max_epochs']
        else:
            lr = config['lr']
            patience = config['patience']
            max_epochs = config.get('max_epochs', 200)
        
        info_text = (
            f"Seed: {seed}\n"
            f"LR: {lr}\n"
            f"Max Epochs: {max_epochs}\n"
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
    
    # Extract hyperparameters from first seed for filename
    first_seed = seeds[0]
    log_file = Path(config['runs_dir']) / dataset_name / model_name / f'seed_{first_seed}' / f'K_{K}' / 'train_log.csv'
    df_first = pd.read_csv(log_file)
    if 'lr' in df_first.columns:
        lr_val = df_first.loc[0, 'lr']
        patience_val = int(df_first.loc[0, 'patience'])
        max_epochs_val = int(df_first.loc[0, 'max_epochs'])
    else:
        lr_val = config['lr']
        patience_val = config['patience']
        max_epochs_val = config.get('max_epochs', 200)
    
    output_dir = Path(config['figures_dir']) / dataset_name / model_name / f'K_{K}'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{dataset_name}_{model_name}_k{K}_lr{lr_val}_p{patience_val}_e{max_epochs_val}_training_curves.pdf'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"  ✓ Saved: {output_file}")


def plot_accuracy_curves(dataset_name, model_name, K, config, seeds=None):
    """
    Plot training and validation accuracy curves for each seed.
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
        
        # Plot accuracies (train and val only)
        ax.plot(df['epoch'], df['train_acc'], 'o-', label='Train Acc',
                linewidth=2, markersize=3, alpha=0.7, color='#2E86AB')
        ax.plot(df['epoch'], df['val_acc'], 's-', label='Val Acc',
                linewidth=2, markersize=3, alpha=0.7, color='#F18F01')
        
        # Mark best epoch (by val_loss)
        best_epoch = df['val_loss'].idxmin()
        ax.axvline(best_epoch + 1, color='red', linestyle='--', linewidth=2,
                   alpha=0.5, label=f'Best Epoch={best_epoch+1}')
        
        # Show accuracies at best epoch
        train_acc_at_best = df.loc[best_epoch, 'train_acc']
        val_acc_at_best = df.loc[best_epoch, 'val_acc']
        
        # Read hyperparameters
        if 'lr' in df.columns:
            lr = df.loc[0, 'lr']
            patience = df.loc[0, 'patience']
            max_epochs = df.loc[0, 'max_epochs']
        else:
            from pathlib import Path
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            import config as cfg
            lr = config['lr']
            patience = config['patience']
            max_epochs = config.get('max_epochs', 200)
        
        info_text = (
            f"Seed: {seed}\n"
            f"LR: {lr}, MaxEp: {int(max_epochs)}, Pat: {int(patience)}\n"
            f"Train Acc @ Best: {train_acc_at_best:.3f}\n"
            f"Val Acc @ Best: {val_acc_at_best:.3f}"
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
    
    # Extract hyperparameters from first seed for filename
    first_seed = seeds[0]
    log_file = Path(config['runs_dir']) / dataset_name / model_name / f'seed_{first_seed}' / f'K_{K}' / 'train_log.csv'
    df_first = pd.read_csv(log_file)
    if 'lr' in df_first.columns:
        lr_val = df_first.loc[0, 'lr']
        patience_val = int(df_first.loc[0, 'patience'])
        max_epochs_val = int(df_first.loc[0, 'max_epochs'])
    else:
        lr_val = config['lr']
        patience_val = config['patience']
        max_epochs_val = config.get('max_epochs', 200)
    
    output_dir = Path(config['figures_dir']) / dataset_name / model_name / f'K_{K}'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{dataset_name}_{model_name}_k{K}_lr{lr_val}_p{patience_val}_e{max_epochs_val}_accuracy_curves.pdf'
    fig.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"  ✓ Saved: {output_file}")


def has_splits(dataset_name, model_name, K, seed, config):
    """
    Check if this dataset/model/K/seed combination uses split-based structure.
    Returns True if split_0 subdirectory exists, False otherwise.
    """
    runs_dir = Path(config['runs_dir']) / dataset_name / model_name / f'seed_{seed}' / f'K_{K}'
    split_0_dir = runs_dir / 'split_0'
    return split_0_dir.exists()


def plot_per_split_diagnostics(dataset_name, model_name, K, config, seeds=None, num_splits=10):
    """
    Plot combined training diagnostics for datasets with per-split structure.
    Creates one large figure with 2 rows (accuracy, loss) and num_splits columns.
    """
    if seeds is None:
        seeds = config['seeds']
    
    for seed in seeds:
        fig, axes = plt.subplots(2, num_splits, figsize=(3*num_splits, 8))
        
        for split_idx in range(num_splits):
            log_file = Path(config['runs_dir']) / dataset_name / model_name / f'seed_{seed}' / f'K_{K}' / f'split_{split_idx}' / 'train_log.csv'
            
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
            
            ax.text(0.5, 0.98, f'Best: {best_epoch+1}\nNLL: {best_val_loss:.2f}',
                    transform=ax.transAxes, fontsize=8, ha='center',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            ax.set_xlabel('Epoch', fontsize=9)
            if split_idx == 0:
                ax.set_ylabel('Loss (NLL)', fontsize=10)
            if split_idx == 0:
                ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
        
        # Extract hyperparameters first to include in title
        log_file_0 = Path(config['runs_dir']) / dataset_name / model_name / f'seed_{seed}' / f'K_{K}' / 'split_0' / 'train_log.csv'
        df_0 = pd.read_csv(log_file_0)
        if 'lr' in df_0.columns:
            lr_val = df_0.loc[0, 'lr']
            patience_val = int(df_0.loc[0, 'patience'])
            max_epochs_val = int(df_0.loc[0, 'max_epochs'])
        else:
            lr_val = config['lr']
            patience_val = config['patience']
            max_epochs_val = config.get('max_epochs', 200)
        
        # Updated title with hyperparameters
        title = (f'Per-Split Training Diagnostics: {dataset_name} - {model_name} (K={K}, Seed={seed})\n'
                 f'LR={lr_val}, Max Epochs={max_epochs_val}, Patience={patience_val}')
        fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        output_dir = Path(config['figures_dir']) / dataset_name / model_name / f'K_{K}'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'{dataset_name}_{model_name}_k{K}_seed{seed}_lr{lr_val}_p{patience_val}_e{max_epochs_val}_per_split_diagnostics.png'
        fig.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        print(f"  ✓ Saved: {output_file}")


def plot_combined_diagnostics(dataset_name, model_name, K, config, seeds=None):
    """
    Plot combined training diagnostics: accuracy curves (top), loss curves (bottom).
    Creates one 2-row figure for all seeds.
    """
    if seeds is None:
        seeds = config['seeds']
    
    n_seeds = len(seeds)
    fig, axes = plt.subplots(2, n_seeds, figsize=(6*n_seeds, 10))
    
    if n_seeds == 1:
        axes = axes.reshape(2, 1)
    
    for idx, seed in enumerate(seeds):
        log_file = Path(config['runs_dir']) / dataset_name / model_name / f'seed_{seed}' / f'K_{K}' / 'train_log.csv'
        
        if not log_file.exists():
            print(f"  ⚠ Training log not found for seed {seed}")
            continue
        
        df = pd.read_csv(log_file)
        best_epoch = df['val_loss'].idxmin()
        
        # Top row: Accuracy curves
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
        
        # Bottom row: Loss curves
        ax = axes[1, idx]
        ax.plot(df['epoch'], df['train_loss'], 'o-', label='Train Loss',
                linewidth=2, markersize=4, alpha=0.7, color='#2E86AB')
        ax.plot(df['epoch'], df['val_loss'], 's-', label='Val Loss',
                linewidth=2, markersize=4, alpha=0.7, color='#F18F01')
        ax.axvline(best_epoch + 1, color='red', linestyle='--', linewidth=2,
                   alpha=0.5)
        
        best_val_loss = df.loc[best_epoch, 'val_loss']
        best_train_loss = df.loc[best_epoch, 'train_loss']
        # Add info text
        # Try to read hyperparameters from CSV (saved in newer runs)
        # Fall back to config for backward compatibility with older runs
        if 'lr' in df.columns:
            lr = df.loc[0, 'lr']  # Same for all epochs
            patience = df.loc[0, 'patience']
            max_epochs = df.loc[0, 'max_epochs']
        else:
            lr = config['lr']
            patience = config['patience']
            max_epochs = config.get('max_epochs', 200)
        
        info_text = (
            f"LR: {lr}\\n"
            f"Patience: {patience}\\n"
            f"Best Epoch: {best_epoch+1}/{max_epochs}\\n"
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
    
    fig.suptitle(f'Training Diagnostics: {dataset_name} - {model_name} (K={K})',
                 fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Extract hyperparameters from first seed for filename
    first_seed = seeds[0]
    log_file = Path(config['runs_dir']) / dataset_name / model_name / f'seed_{first_seed}' / f'K_{K}' / 'train_log.csv'
    df_first = pd.read_csv(log_file)
    if 'lr' in df_first.columns:
        lr_val = df_first.loc[0, 'lr']
        patience_val = int(df_first.loc[0, 'patience'])
        max_epochs_val = int(df_first.loc[0, 'max_epochs'])
    else:
        lr_val = config['lr']
        patience_val = config['patience']
        max_epochs_val = config.get('max_epochs', 200)
    
    output_dir = Path(config['figures_dir']) / dataset_name / model_name / f'K_{K}'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{dataset_name}_{model_name}_k{K}_lr{lr_val}_p{patience_val}_e{max_epochs_val}_training_diagnostics.png'
    fig.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"  ✓ Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate training diagnostic plots')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--K', type=int, required=True,
                       help='K value (model depth)')
    parser.add_argument('--num-splits', type=int, default=10,
                       help='Number of splits for heterophilous datasets (default: 10)')
    
    args = parser.parse_args()
    
    # Convert config module to dict
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    print(f"\n{'='*70}")
    print(f"Generating Training Diagnostic Plots: {args.dataset} {args.model} (K={args.K})")
    print(f"{'='*70}\n")
    
    # Detect if dataset uses split-based structure
    first_seed = config['seeds'][0] if config['seeds'] else 0
    uses_splits = has_splits(args.dataset, args.model, args.K, first_seed, config)
    
    if uses_splits:
        print(f"  Detected per-split structure. Generating diagnostics for {args.num_splits} splits...")
        plot_per_split_diagnostics(args.dataset, args.model, args.K, config, num_splits=args.num_splits)
    else:
        print("  Generating combined training diagnostics...")
        plot_combined_diagnostics(args.dataset, args.model, args.K, config)
    
    print(f"\n{'='*60}")
    print("✓ Training diagnostic plots complete!")
    print(f"{'='*60}\n")



if __name__ == '__main__':
    main()
