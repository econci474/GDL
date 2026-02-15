"""Compute separability metrics from classifier head outputs.

This script computes AUROC and Cohen's d for error detection based on entropy,
directly from classifier head layer_probs.npz outputs.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

try:
    from torchmetrics import AUROC
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics not available. Install with: pip install torchmetrics")


def compute_auroc_torchmetrics(H, e):
    """
    Compute AUROC for error detection using entropy as the score.
    
    Args:
        H: torch.Tensor of entropy scores [N]
        e: torch.Tensor of error indicators (1=wrong, 0=correct) [N]
        
    Returns:
        auroc: AUROC score or NaN if undefined
    """
    if not TORCHMETRICS_AVAILABLE:
        return np.nan
    
    # Convert to tensors if needed
    if isinstance(H, np.ndarray):
        H = torch.from_numpy(H).float()
    if isinstance(e, np.ndarray):
        e = torch.from_numpy(e).long()
    
    # Check for edge cases
    unique_labels = torch.unique(e)
    if len(unique_labels) < 2:
        # All correct or all incorrect
        return np.nan
    
    try:
        # Use torchmetrics AUROC
        auroc_fn = AUROC(task='binary')
        auroc_score = auroc_fn(H, e).item()
        return auroc_score
    except Exception as ex:
        print(f"Warning: AUROC computation failed: {ex}")
        return np.nan


def compute_cohens_d(H_wrong, H_correct):
    """
    Compute Cohen's d effect size for entropy separation.
    
    d = (mean_wrong - mean_correct) / pooled_std
    
    Args:
        H_wrong: Entropy values for incorrect predictions
        H_correct: Entropy values for correct predictions
        
    Returns:
        cohens_d: Effect size or NaN if not computable
    """
    n_w = len(H_wrong)
    n_c = len(H_correct)
    
    # Need at least 2 samples in each group
    if n_w < 2 or n_c < 2:
        return np.nan
    
    mean_w = np.mean(H_wrong)
    mean_c = np.mean(H_correct)
    
    var_w = np.var(H_wrong, ddof=1)
    var_c = np.var(H_correct, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n_w - 1) * var_w + (n_c - 1) * var_c) / (n_w + n_c - 2))
    
    if pooled_std == 0:
        return np.nan
    
    cohens_d = (mean_w - mean_c) / pooled_std
    return cohens_d


def calc_entropy(probs):
    """Calculate entropy per node."""
    return -np.sum(probs * np.log(probs + 1e-10), axis=1)


def compute_separability_from_classifier_outputs(dataset, model, K, seed, loss_type, split='val'):
    """
    Compute separability metrics from classifier head outputs.
    
    Args:
        dataset: Dataset name
        model: Model name
        K: Number of layers
        seed: Random seed
        loss_type: Loss type directory name
        split: 'val' or 'test'
    
    Returns:
        DataFrame with metrics for each layer
    """
    # Load the dataset to get labels
    from src.datasets import load_dataset as load_ds
    data_obj, num_classes, _ = load_ds(
        dataset,
        root_dir='data',
        planetoid_normalize=False,
        planetoid_split='public'
    )
    
    labels = data_obj.y.numpy()
    if split == 'val':
        split_mask = data_obj.val_mask.numpy()
    else:
        split_mask = data_obj.test_mask.numpy()
    
    labels_split = labels[split_mask]
    
    # Load classifier outputs
    probs_path = Path(cfg.classifier_heads_dir) / loss_type / dataset / model / f'seed_{seed}' / f'K_{K}' / 'layer_probs.npz'
    if not probs_path.exists():
        raise FileNotFoundError(f"Classifier outputs not found: {probs_path}")
    
    probs_data = np.load(probs_path)
    
    # Compute metrics for each layer
    results = []
    per_class_metrics = []  # Store per-class data for plotting
    
    for k in range(K + 1):
        # Load probabilities for this layer
        probs_k = probs_data[f'{split}_probs_{k}']
        
        # Compute predictions
        preds_k = probs_k.argmax(axis=1)
        
        # Compute correctness
        correct = (preds_k == labels_split).astype(int)
        errors = 1 - correct
        
        # Compute entropy
        entropy_k = calc_entropy(probs_k)
        
        # Compute AUROC
        auroc = compute_auroc_torchmetrics(entropy_k, errors)
        
        # Compute Cohen's d
        H_correct = entropy_k[correct == 1]
        H_wrong = entropy_k[correct == 0]
        cohens_d = compute_cohens_d(H_wrong, H_correct)
        
        # Compute accuracy
        accuracy = correct.mean()
        
        # Compute mean entropy
        mean_entropy = entropy_k.mean()
        mean_entropy_correct = H_correct.mean() if len(H_correct) > 0 else np.nan
        mean_entropy_wrong = H_wrong.mean() if len(H_wrong) > 0 else np.nan
        
        # Per-class metrics
        per_class_data = {'k': k}
        for c in range(num_classes):
            class_mask = (labels_split == c)
            n_class_total = class_mask.sum()
            
            if n_class_total > 0:
                class_probs = probs_k[class_mask]
                class_preds = preds_k[class_mask]
                class_correct = (class_preds == c).astype(int)
                class_entropy = calc_entropy(class_probs)
                
                n_class_correct = class_correct.sum()
                n_class_wrong = n_class_total - n_class_correct
                
                # Per-class accuracy
                class_accuracy = class_correct.mean()
                
                # Per-class Cohen's d (requires at least 2 samples in each group)
                H_c_correct = class_entropy[class_correct == 1]
                H_c_wrong = class_entropy[class_correct == 0]
                class_cohens_d = compute_cohens_d(H_c_wrong, H_c_correct) if len(H_c_correct) >= 2 and len(H_c_wrong) >= 2 else np.nan
                
                # Per-class mean entropy
                class_mean_entropy = class_entropy.mean()
                
                per_class_data[f'class_{c}_accuracy'] = class_accuracy
                per_class_data[f'class_{c}_cohens_d'] = class_cohens_d
                per_class_data[f'class_{c}_entropy'] = class_mean_entropy
                per_class_data[f'class_{c}_n_total'] = n_class_total
                per_class_data[f'class_{c}_n_correct'] = n_class_correct
                per_class_data[f'class_{c}_n_wrong'] = n_class_wrong
            else:
                per_class_data[f'class_{c}_accuracy'] = np.nan
                per_class_data[f'class_{c}_cohens_d'] = np.nan
                per_class_data[f'class_{c}_entropy'] = np.nan
                per_class_data[f'class_{c}_n_total'] = 0
                per_class_data[f'class_{c}_n_correct'] = 0
                per_class_data[f'class_{c}_n_wrong'] = 0
        
        per_class_metrics.append(per_class_data)
        
        results.append({
            'k': k,
            'accuracy': accuracy,
            'auroc': auroc,
            'cohens_d': cohens_d,
            'mean_entropy': mean_entropy,
            'mean_entropy_correct': mean_entropy_correct,
            'mean_entropy_wrong': mean_entropy_wrong,
            'n_correct': len(H_correct),
            'n_wrong': len(H_wrong)
        })
    
    return pd.DataFrame(results), pd.DataFrame(per_class_metrics), num_classes


def plot_separability_vs_k(df, df_per_class, num_classes, dataset, model, K, seed, loss_type, output_dir):
    """Plot separability metrics vs layer depth with per-class analysis."""
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle(f'Separability Metrics: {dataset} {model} K={K} seed={seed}\n{loss_type}',
                 fontsize=14, fontweight='bold')
    
    layers = df['k'].values
    
    # Plot 1: Accuracy
    ax = axes[0, 0]
    ax.plot(layers, df['accuracy'], 'o-', linewidth=2, markersize=8, color='C0')
    ax.set_xlabel('Layer Depth (k)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Validation Accuracy by Layer', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 2: AUROC
    ax = axes[0, 1]
    ax.plot(layers, df['auroc'], 's-', linewidth=2, markersize=8, color='C1')
    ax.set_xlabel('Layer Depth (k)', fontsize=11, fontweight='bold')
    ax.set_ylabel('AUROC', fontsize=11, fontweight='bold')
    ax.set_title('AUROC for Error Detection', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.legend()
    
    # Plot 3: Cohen's d
    ax = axes[1, 0]
    ax.plot(layers, df['cohens_d'], '^-', linewidth=2, markersize=8, color='C2')
    ax.set_xlabel('Layer Depth (k)', fontsize=11, fontweight='bold')
    ax.set_ylabel("Cohen's d", fontsize=11, fontweight='bold')
    ax.set_title("Cohen's d (Effect Size)", fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 4: Mean entropy (correct vs wrong)
    ax = axes[1, 1]
    ax.plot(layers, df['mean_entropy_correct'], 'o-', linewidth=2, markersize=6, 
            label='Correct', color='green', alpha=0.7)
    ax.plot(layers, df['mean_entropy_wrong'], 's-', linewidth=2, markersize=6,
            label='Wrong', color='red', alpha=0.7)
    ax.set_xlabel('Layer Depth (k)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Entropy', fontsize=11, fontweight='bold')
    ax.set_title('Mean Entropy: Correct vs Wrong', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Plot 5: Per-class validation accuracy
    ax = axes[2, 0]
    colors = plt.cm.tab10(np.arange(num_classes))
    for c in range(num_classes):
        accuracy_col = f'class_{c}_accuracy'
        if accuracy_col in df_per_class.columns:
            ax.plot(df_per_class['k'], df_per_class[accuracy_col], 
                   'o-', linewidth=1.5, markersize=5, label=f'Class {c}', color=colors[c])
    ax.set_xlabel('Layer Depth (k)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Per-Class Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Per-Class Validation Accuracy by Layer', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.05])
    # No legend for clarity
    
    # Plot 6: Per-class mean entropy (with detailed legend)
    ax = axes[2, 1]
    
    # Get node counts from last layer (k=K) for legend
    last_layer_idx = df_per_class[df_per_class['k'] == K].index[0]
    
    # Sort classes by total node count for legend ordering
    class_sizes = []
    for c in range(num_classes):
        n_total = int(df_per_class.loc[last_layer_idx, f'class_{c}_n_total'])
        class_sizes.append((c, n_total))
    
    # Sort by n_total (ascending)
    class_sizes.sort(key=lambda x: x[1])
    sorted_classes = [c for c, _ in class_sizes]
    
    for c in sorted_classes:
        entropy_col = f'class_{c}_entropy'
        if entropy_col in df_per_class.columns:
            # Get counts for this class
            n_total = int(df_per_class.loc[last_layer_idx, f'class_{c}_n_total'])
            n_correct = int(df_per_class.loc[last_layer_idx, f'class_{c}_n_correct'])
            n_wrong = int(df_per_class.loc[last_layer_idx, f'class_{c}_n_wrong'])
            
            label = f'C{c} (n={n_total}: {n_correct}✓/{n_wrong}✗)'
            ax.plot(df_per_class['k'], df_per_class[entropy_col],
                   'o-', linewidth=1.5, markersize=5, label=label, color=colors[c])
    
    ax.set_xlabel('Layer Depth (k)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Per-Class Mean Entropy', fontsize=11, fontweight='bold')
    ax.set_title('Per-Class Mean Entropy by Layer', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, ncol=1, loc='best')
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f'{dataset}_{model}_k{K}_seed{seed}_separability_{loss_type}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'✓ Separability plot saved to: {output_path}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compute separability metrics from classifier head outputs')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--loss-type', type=str, required=True,
                       help='Loss type directory name (e.g., ce_plus_R_R1.0_hard)')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    
    args = parser.parse_args()
    
    print(f'\n{"="*60}')
    print(f'Separability Metrics (Classifier Heads): {args.model} on {args.dataset}')
    print(f'  K={args.K}, seed={args.seed}, loss_type={args.loss_type}, split={args.split}')
    print(f'{"="*60}\n')
    
    # Compute metrics
    print('Computing separability metrics...')
    df, df_per_class, num_classes = compute_separability_from_classifier_outputs(
        args.dataset, args.model, args.K, args.seed, args.loss_type, args.split
    )
    
    print('\nResults:')
    print(df.to_string(index=False))
    
    # Save CSV
    output_dir = Path(cfg.tables_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f'{args.dataset}_{args.model}_K{args.K}_seed{args.seed}_{args.loss_type}_separability.csv'
    df.to_csv(csv_path, index=False)
    print(f'\n✓ Results saved to: {csv_path}')
    
    # Generate plot
    plot_dir = Path(cfg.figures_dir) / args.dataset / args.model / f'K_{args.K}'
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_separability_vs_k(df, df_per_class, num_classes, args.dataset, args.model, args.K, args.seed, args.loss_type, plot_dir)
    
    print(f'\n{"="*60}')
    print('✓ Separability analysis complete')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
