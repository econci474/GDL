"""Compute separability metrics and constrained depth selection."""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

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
    
    Uses validation set only for all selection decisions (offline probing).
    
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
    
    std_w = np.std(H_wrong, ddof=1)
    std_c = np.std(H_correct, ddof=1)
    
    # Pooled standard deviation
    s_pooled = np.sqrt(((n_w - 1) * std_w**2 + (n_c - 1) * std_c**2) / (n_w + n_c - 2))
    
    if s_pooled == 0:
        return np.nan
    
    d = (mean_w - mean_c) / s_pooled
    
    return d


def compute_entropy_auc(H_correct, H_incorrect):
    """
    Compute area under the entropy curve for correct and incorrect predictions.
    
    This is simply the mean entropy for each group.
    
    Args:
        H_correct: Entropy values for correct predictions
        H_incorrect: Entropy values for incorrect predictions
        
    Returns:
        auc_correct: Mean entropy for correct predictions
        auc_incorrect: Mean entropy for incorrect predictions
    """
    auc_correct = np.mean(H_correct) if len(H_correct) > 0 else np.nan
    auc_incorrect = np.mean(H_incorrect) if len(H_incorrect) > 0 else np.nan
    
    return auc_correct, auc_incorrect


def compute_separability_metrics_at_depth(H_val, e_val, H_test, e_test):
    """
    Compute all separability metrics for a single depth k.
    
    Validation set only is used for selection decisions (offline probing).
    
    Args:
        H_val: Validation entropy scores
        e_val: Validation error indicators
        H_test: Test entropy scores
        e_test: Test error indicators
        
    Returns:
        dict with metrics
    """
    # Validation metrics (used for selection)
    val_auroc = compute_auroc_torchmetrics(H_val, e_val)
    
    H_val_correct = H_val[e_val == 0]
    H_val_wrong = H_val[e_val == 1]
    
    val_cohens_d = compute_cohens_d(H_val_wrong, H_val_correct)
    val_auc_correct, val_auc_incorrect = compute_entropy_auc(H_val_correct, H_val_wrong)
    
    # Test metrics (for reporting only)
    test_auroc = compute_auroc_torchmetrics(H_test, e_test)
    
    H_test_correct = H_test[e_test == 0]
    H_test_wrong = H_test[e_test == 1]
    
    test_cohens_d = compute_cohens_d(H_test_wrong, H_test_correct)
    test_auc_correct, test_auc_incorrect = compute_entropy_auc(H_test_correct, H_test_wrong)
    
    metrics = {
        'val_auroc_err_from_entropy': val_auroc,
        'val_cohens_d': val_cohens_d,
        'val_n_wrong': len(H_val_wrong),
        'val_n_correct': len(H_val_correct),
        'val_entropy_auc_correct': val_auc_correct,
        'val_entropy_auc_incorrect': val_auc_incorrect,
        'test_auroc_err_from_entropy': test_auroc,
        'test_cohens_d': test_cohens_d,
        'test_entropy_auc_correct': test_auc_correct,
        'test_entropy_auc_incorrect': test_auc_incorrect,
    }
    
    return metrics


def select_k_star_constrained(df, eps_acc=0.01):
    """
    Select optimal depth k* using constrained separability maximization.
    
    Validation-only selection (offline probing).
    
    Constraint: val_acc >= max(val_acc) - eps_acc
    Objective: maximize AUROC (fallback: Cohen's d, then NLL)
    
    Args:
        df: DataFrame with columns: k, val_acc, val_auroc_err_from_entropy, val_cohens_d, val_nll
        eps_acc: Accuracy tolerance (default 0.01 = 1 percentage point)
        
    Returns:
        k_star_sep: Selected depth
        k_star_method: Selection method used
        k_best_val_acc: Depth with best validation accuracy
    """
    # Find best validation accuracy
    best_val_acc = df['val_acc'].max()
    k_best_val_acc = df.loc[df['val_acc'].idxmax(), 'k']
    
    # Filter candidate set
    acc_threshold = best_val_acc - eps_acc
    candidates = df[df['val_acc'] >= acc_threshold].copy()
    
    print(f"\n  Best val acc: {best_val_acc:.4f} at k={int(k_best_val_acc)}")
    print(f"  Accuracy threshold: {acc_threshold:.4f}")
    print(f"  Candidate depths: {candidates['k'].tolist()}")
    
    # Try AUROC first
    auroc_values = candidates['val_auroc_err_from_entropy'].dropna()
    if len(auroc_values) > 0:
        k_star_sep = candidates.loc[auroc_values.idxmax(), 'k']
        k_star_method = 'auroc'
        print(f"  Selected k*={int(k_star_sep)} via AUROC")
        return int(k_star_sep), k_star_method, int(k_best_val_acc)
    
    # Fallback to Cohen's d
    cohens_d_values = candidates['val_cohens_d'].dropna()
    if len(cohens_d_values) > 0:
        k_star_sep = candidates.loc[cohens_d_values.idxmax(), 'k']
        k_star_method = 'cohens_d'
        print(f"  Selected k*={int(k_star_sep)} via Cohen's d")
        return int(k_star_sep), k_star_method, int(k_best_val_acc)
    
    # Fallback to NLL
    nll_values = candidates['val_nll'].dropna()
    if len(nll_values) > 0:
        k_star_sep = candidates.loc[nll_values.idxmin(), 'k']
        k_star_method = 'val_nll_fallback'
        print(f"  Selected k*={int(k_star_sep)} via NLL fallback")
        return int(k_star_sep), k_star_method, int(k_best_val_acc)
    
    # Ultimate fallback: best accuracy
    k_star_sep = k_best_val_acc
    k_star_method = 'val_acc_fallback'
    print(f"  Selected k*={int(k_star_sep)} via accuracy fallback")
    return int(k_star_sep), k_star_method, int(k_best_val_acc)


def select_k_star_nll(df):
    """
    Select optimal depth using validation NLL (baseline method).
    
    This is the standard depth selection from depth_selection.py.
    
    Args:
        df: DataFrame with columns: k, val_nll
        
    Returns:
        k_star_nll: Depth with minimum validation NLL
    """
    k_star_nll = df.loc[df['val_nll'].idxmin(), 'k']
    return int(k_star_nll)


def select_k_star_combined(df, lambda_val=0.1):
    """
    Select optimal depth using combined NLL + λ*entropy (baseline method).
    
    This is the combined depth selection from depth_selection.py.
    
    Args:
        df: DataFrame with columns: k, val_nll, val_entropy_mean
        lambda_val: Weight for entropy term (default 0.1)
        
    Returns:
        k_star_combined: Depth minimizing val_nll + λ*val_entropy_mean
    """
    df = df.copy()
    df['combined_score'] = df['val_nll'] + lambda_val * df['val_entropy_mean']
    k_star_combined = df.loc[df['combined_score'].idxmin(), 'k']
    return int(k_star_combined)


def select_k_star_top3(df):
    """
    Select optimal depth from top-3 validation accuracy depths.
    
    This method:
    1. Identifies the 3 depths with highest validation accuracy
    2. Maximizes AUROC within that top-3 set
    3. Fallback to Cohen's d if AUROC unavailable
    
    Args:
        df: DataFrame with columns: k, val_acc, val_auroc_err_from_entropy, val_cohens_d
        
    Returns:
        k_star_top3: Depth maximizing separability among top-3 accuracy
    """
    # Get top 3 depths by validation accuracy
    top3_indices = df.nlargest(3, 'val_acc').index
    candidates = df.loc[top3_indices].copy()
    
    # Try AUROC first
    auroc_values = candidates['val_auroc_err_from_entropy'].dropna()
    if len(auroc_values) > 0:
        k_star_top3 = candidates.loc[auroc_values.idxmax(), 'k']
        return int(k_star_top3)
    
    # Fallback to Cohen's d
    cohens_d_values = candidates['val_cohens_d'].dropna()
    if len(cohens_d_values) > 0:
        k_star_top3 = candidates.loc[cohens_d_values.idxmax(), 'k']
        return int(k_star_top3)
    
    # Ultimate fallback: best accuracy in top-3
    k_star_top3 = candidates.loc[candidates['val_acc'].idxmax(), 'k']
    return int(k_star_top3)


def compute_spearman_correlations(df):
    """
    Compute Spearman rank correlations across depth k.
    
    Args:
        df: DataFrame with k and various metrics
        
    Returns:
        dict of correlation coefficients
    """
    correlations = {}
    
    # Correlations to compute
    metrics = ['val_entropy_mean', 'val_acc', 'val_nll', 'val_auroc_err_from_entropy']
    
    for metric in metrics:
        if metric in df.columns:
            # Drop NaN values for correlation
            valid = df[['k', metric]].dropna()
            if len(valid) > 2:
                rho, pval = spearmanr(valid['k'], valid[metric])
                correlations[f'rho_k_{metric}'] = rho
                correlations[f'pval_k_{metric}'] = pval
            else:
                correlations[f'rho_k_{metric}'] = np.nan
                correlations[f'pval_k_{metric}'] = np.nan
    
    return correlations


def plot_separability_vs_depth(df, df_per_class, num_classes, summary, output_path):
    """
    Plot separability metrics vs depth with per-class analysis.
    
    Args:
        df: DataFrame with aggregate metrics
        df_per_class: DataFrame with per-class metrics
        num_classes: Number of classes
        summary: Summary dict with k_star values
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    
    k_values = df['k'].values
    
    # Panel 1: AUROC vs k
    ax = axes[0, 0]
    ax.plot(k_values, df['val_auroc_err_from_entropy'], 'o-', label='Validation AUROC', color='tab:blue', linewidth=2, markersize=6)
    ax.set_xlabel('Depth k', fontsize=11, fontweight='bold')
    ax.set_ylabel('AUROC', fontsize=11, fontweight='bold')
    ax.set_title('Error Detection AUROC vs Depth', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    
    # Panel 2: Cohen's d vs k
    ax = axes[0, 1]
    ax.plot(k_values, df['val_cohens_d'], 'o-', label="Cohen's d", color='tab:orange', linewidth=2, markersize=6)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Depth k', fontsize=11, fontweight='bold')
    ax.set_ylabel("Cohen's d", fontsize=11, fontweight='bold')
    ax.set_title("Entropy Separability (Cohen's d) vs Depth", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Validation accuracy vs k
    ax = axes[1, 0]
    ax.plot(k_values, df['val_acc'], 'o-', label='Validation Accuracy', color='tab:green', linewidth=2, markersize=6)
    ax.set_xlabel('Depth k', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Validation Accuracy vs Depth', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Panel 4: Entropy AUC for correct vs incorrect
    ax = axes[1, 1]
    ax.plot(k_values, df['val_entropy_auc_correct'], 'o-', label='Correct Predictions', color='green', linewidth=2, markersize=6, alpha=0.7)
    ax.plot(k_values, df['val_entropy_auc_incorrect'], 's-', label='Incorrect Predictions', color='red', linewidth=2, markersize=6, alpha=0.7)
    ax.set_xlabel('Depth k', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Entropy', fontsize=11, fontweight='bold')
    ax.set_title('Mean Entropy: Correct vs Incorrect', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Panel 5: Per-class validation accuracy
    ax = axes[2, 0]
    colors = plt.cm.tab10(np.arange(num_classes))
    for c in range(num_classes):
        accuracy_col = f'class_{c}_accuracy'
        if accuracy_col in df_per_class.columns:
            ax.plot(df_per_class['k'], df_per_class[accuracy_col], 
                   'o-', linewidth=1.5, markersize=5, label=f'Class {c}', color=colors[c])
    ax.set_xlabel('Depth k', fontsize=11, fontweight='bold')
    ax.set_ylabel('Per-Class Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Per-Class Validation Accuracy by Depth', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    # No legend for clarity with many classes
    
    # Panel 6: Per-class mean entropy (with detailed legend)
    ax = axes[2, 1]
    
    # Get node counts from last layer for legend
    last_k = df_per_class['k'].max()
    last_layer_idx = df_per_class[df_per_class['k'] == last_k].index[0]
    
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
    
    ax.set_xlabel('Depth k', fontsize=11, fontweight='bold')
    ax.set_ylabel('Per-Class Mean Entropy', fontsize=11, fontweight='bold')
    ax.set_title('Per-Class Mean Entropy by Depth', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=1, loc='best')
    
    # Mark special k values with vertical lines on first subplot
    ax_first = axes[0, 0]
    
    # k_star_nll (baseline NLL method) - offset left
    if 'k_star_nll' in summary:
        ax_first.axvline(x=summary['k_star_nll'] - 0.1, color='purple', linestyle='-.', 
                   alpha=0.6, linewidth=1.5, label=f'k*_nll={summary["k_star_nll"]}')
    # k_star_combined (NLL + λ*entropy method) - offset right
    if 'k_star_combined' in summary:
        ax_first.axvline(x=summary['k_star_combined'] + 0.1, color='orange', linestyle=':', 
                   alpha=0.6, linewidth=1.5, label=f'k*_combined={summary["k_star_combined"]}')
    # k_star_top3 (top-3 accuracy, max AUROC) - no offset
    if 'k_star_top3' in summary:
        ax_first.axvline(x=summary['k_star_top3'], color='cyan', linestyle='-', 
                   alpha=0.6, linewidth=1.5, label=f'k*_top3={summary["k_star_top3"]}')
    # k_best_val_acc (best accuracy) - offset left
    ax_first.axvline(x=summary['k_best_val_acc'] - 0.1, color='green', linestyle=':', 
               alpha=0.5, label=f'k_best_acc={summary["k_best_val_acc"]}')
    # k_star_sep (constrained separability method) - offset right
    ax_first.axvline(x=summary['k_star_sep'] + 0.1, color='red', linestyle='--', 
               alpha=0.7, linewidth=2, label=f'k*_sep={summary["k_star_sep"]}')
    
    # Add the same lines to all other subplots (without labels to avoid duplicates)
    for ax in axes.flat:
        if ax != ax_first:
            if 'k_star_nll' in summary:
                ax.axvline(x=summary['k_star_nll'] - 0.1, color='purple', linestyle='-.', 
                          alpha=0.6, linewidth=1.5)
            if 'k_star_combined' in summary:
                ax.axvline(x=summary['k_star_combined'] + 0.1, color='orange', linestyle=':', 
                          alpha=0.6, linewidth=1.5)
            if 'k_star_top3' in summary:
                ax.axvline(x=summary['k_star_top3'], color='cyan', linestyle='-', 
                          alpha=0.6, linewidth=1.5)
            ax.axvline(x=summary['k_best_val_acc'] - 0.1, color='green', linestyle=':', 
                      alpha=0.5)
            ax.axvline(x=summary['k_star_sep'] + 0.1, color='red', linestyle='--', 
                      alpha=0.7, linewidth=2)
    
    # Update legend on first subplot to show all items
    ax_first.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Plot saved to: {output_path}")


def plot_aggregated_seeds(dataset, model, K, seeds, config):
    """
    Create aggregated plot across multiple seeds showing mean ± std.
    
    Args:
        dataset: Dataset name
        model: Model name
        K: Maximum depth
        seeds: List of seed values
        config: Config dictionary
    """
    # Load enriched CSVs for all seeds
    all_dfs = []
    for seed in seeds:
        csv_path = Path(config['tables_dir']) / f'{dataset}_{model}_K{K}_seed{seed}_probe_with_separability.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['seed'] = seed
            all_dfs.append(df)
        else:
            print(f"Warning: Missing data for seed {seed}, skipping")
    
    if len(all_dfs) == 0:
        print("No data found for aggregation")
        return
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Compute mean and std for each k
    metrics = ['val_auroc_err_from_entropy', 'val_cohens_d', 'val_acc', 
               'val_entropy_auc_correct', 'val_entropy_auc_incorrect']
    
    agg_df = combined_df.groupby('k')[metrics].agg(['mean', 'std']).reset_index()
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    k_values = agg_df['k'].values
    
    # Panel 1: AUROC vs k
    ax = axes[0, 0]
    mean = agg_df[('val_auroc_err_from_entropy', 'mean')].values
    std = agg_df[('val_auroc_err_from_entropy', 'std')].values
    ax.plot(k_values, mean, 'o-', label=f'Mean (n={len(seeds)} seeds)', color='tab:blue')
    ax.fill_between(k_values, mean - std, mean + std, alpha=0.2, color='tab:blue')
    ax.set_xlabel('Depth k')
    ax.set_ylabel('AUROC')
    ax.set_title('Error Detection AUROC vs Depth')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Panel 2: Cohen's d vs k
    ax = axes[0, 1]
    mean = agg_df[('val_cohens_d', 'mean')].values
    std = agg_df[('val_cohens_d', 'std')].values
    ax.plot(k_values, mean, 'o-', label=f'Mean (n={len(seeds)} seeds)', color='tab:orange')
    ax.fill_between(k_values, mean - std, mean + std, alpha=0.2, color='tab:orange')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Depth k')
    ax.set_ylabel("Cohen's d")
    ax.set_title("Entropy Separability (Cohen's d) vs Depth")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Panel 3: Validation accuracy vs k
    ax = axes[1, 0]
    mean = agg_df[('val_acc', 'mean')].values
    std = agg_df[('val_acc', 'std')].values
    ax.plot(k_values, mean, 'o-', label=f'Mean (n={len(seeds)} seeds)', color='tab:green')
    ax.fill_between(k_values, mean - std, mean + std, alpha=0.2, color='tab:green')
    ax.set_xlabel('Depth k')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy vs Depth')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Panel 4: Entropy for correct vs incorrect
    ax = axes[1, 1]
    mean_correct = agg_df[('val_entropy_auc_correct', 'mean')].values
    std_correct = agg_df[('val_entropy_auc_correct', 'std')].values
    mean_incorrect = agg_df[('val_entropy_auc_incorrect', 'mean')].values
    std_incorrect = agg_df[('val_entropy_auc_incorrect', 'std')].values
    
    ax.plot(k_values, mean_correct, 'o-', label='Correct Predictions', color='tab:blue')
    ax.fill_between(k_values, mean_correct - std_correct, mean_correct + std_correct, alpha=0.2, color='tab:blue')
    ax.plot(k_values, mean_incorrect, 's-', label='Incorrect Predictions', color='tab:red')
    ax.fill_between(k_values, mean_incorrect - std_incorrect, mean_incorrect + std_incorrect, alpha=0.2, color='tab:red')
    ax.set_xlabel('Depth k')
    ax.set_ylabel('Mean Entropy')
    ax.set_title('Entropy: Correct vs Incorrect')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Load summary CSVs to check if k* values are unanimous across seeds
    summary_dfs = []
    for seed in seeds:
        summary_path = Path(config['tables_dir']) / f'{dataset}_{model}_K{K}_seed{seed}_separability_summary.csv'
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            summary_dfs.append(summary_df)
    
    if len(summary_dfs) > 0:
        # Concatenate summaries
        all_summaries = pd.concat(summary_dfs, ignore_index=True)
        
        # Check if each k* is unanimous (all seeds agree)
        k_star_nll_values = all_summaries['k_star_nll'].unique()
        k_star_combined_values = all_summaries['k_star_combined'].unique()
        k_star_top3_values = all_summaries['k_star_top3'].unique()
        k_star_sep_values = all_summaries['k_star_sep'].unique()
        k_best_acc_values = all_summaries['k_best_val_acc'].unique()
        
        # Only add lines if unanimous
        ax_first = axes[0, 0]
        
        if len(k_star_nll_values) == 1:
            k_nll = int(k_star_nll_values[0])
            ax_first.axvline(x=k_nll - 0.1, color='purple', linestyle='-.', 
                       alpha=0.6, linewidth=1.5, label=f'k*_nll={k_nll}')
            print(f"  k_star_nll unanimous: {k_nll}")
        else:
            print(f"  k_star_nll not unanimous: {k_star_nll_values.tolist()}")
        
        if len(k_star_combined_values) == 1:
            k_comb = int(k_star_combined_values[0])
            ax_first.axvline(x=k_comb + 0.1, color='orange', linestyle=':', 
                       alpha=0.6, linewidth=1.5, label=f'k*_combined={k_comb}')
            print(f"  k_star_combined unanimous: {k_comb}")
        else:
            print(f"  k_star_combined not unanimous: {k_star_combined_values.tolist()}")
        
        if len(k_star_top3_values) == 1:
            k_top3 = int(k_star_top3_values[0])
            ax_first.axvline(x=k_top3, color='cyan', linestyle='-', 
                       alpha=0.6, linewidth=1.5, label=f'k*_top3={k_top3}')
            print(f"  k_star_top3 unanimous: {k_top3}")
        else:
            print(f"  k_star_top3 not unanimous: {k_star_top3_values.tolist()}")
        
        if len(k_star_sep_values) == 1:
            k_sep = int(k_star_sep_values[0])
            ax_first.axvline(x=k_sep + 0.1, color='red', linestyle='--', 
                       alpha=0.7, linewidth=2, label=f'k*_sep={k_sep}')
            print(f"  k_star_sep unanimous: {k_sep}")
        else:
            print(f"  k_star_sep not unanimous: {k_star_sep_values.tolist()}")
        
        if len(k_best_acc_values) == 1:
            k_acc = int(k_best_acc_values[0])
            ax_first.axvline(x=k_acc - 0.1, color='green', linestyle=':', 
                       alpha=0.5, label=f'k_best_acc={k_acc}')
            print(f"  k_best_acc unanimous: {k_acc}")
        else:
            print(f"  k_best_acc not unanimous: {k_best_acc_values.tolist()}")
        
        # Add same lines to other subplots (without labels)
        for ax in axes.flat:
            if ax != ax_first:
                if len(k_star_nll_values) == 1:
                    ax.axvline(x=int(k_star_nll_values[0]) - 0.1, color='purple', linestyle='-.', 
                              alpha=0.6, linewidth=1.5)
                if len(k_star_combined_values) == 1:
                    ax.axvline(x=int(k_star_combined_values[0]) + 0.1, color='orange', linestyle=':', 
                              alpha=0.6, linewidth=1.5)
                if len(k_star_top3_values) == 1:
                    ax.axvline(x=int(k_star_top3_values[0]), color='cyan', linestyle='-', 
                              alpha=0.6, linewidth=1.5)
                if len(k_star_sep_values) == 1:
                    ax.axvline(x=int(k_star_sep_values[0]) + 0.1, color='red', linestyle='--', 
                              alpha=0.7, linewidth=2)
                if len(k_best_acc_values) == 1:
                    ax.axvline(x=int(k_best_acc_values[0]) - 0.1, color='green', linestyle=':', 
                              alpha=0.5)
        
        # Update legend on first subplot
        ax_first.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    
    # Save to hierarchical directory
    plot_dir = Path(config['figures_dir']) / dataset / model / f'K_{K}'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = plot_dir / f'{dataset}_{model}_k{K}_seed_all_separability_vs_k.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Aggregated plot saved to: {output_path}")
    print(f"  Aggregated across {len(seeds)} seeds")



def main():

    parser = argparse.ArgumentParser(description='Compute separability metrics and constrained depth selection')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--seed', type=str, default='0',
                       help='Seed value or "all" to run all seeds from config')
    parser.add_argument('--eps_acc', type=float, default=0.02,
                       help='Accuracy tolerance for constrained selection (default: 0.02)')
    
    args = parser.parse_args()
    
    # Convert config module to dict
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    # Handle seed argument
    if args.seed.lower() == 'all':
        seeds_to_run = config['seeds']
        run_all_seeds = True
        print(f"\n{'='*60}")
        print(f"Separability Metrics: {args.model} on {args.dataset}")
        print(f"K={args.K}, running all seeds: {seeds_to_run}, eps_acc={args.eps_acc}")
        print(f"{'='*60}")
    else:
        seeds_to_run = [int(args.seed)]
        run_all_seeds = False
        print(f"\n{'='*60}")
        print(f"Separability Metrics: {args.model} on {args.dataset}")
        print(f"K={args.K}, seed={args.seed}, eps_acc={args.eps_acc}")
        print(f"{'='*60}")
    
    
    # Process each seed
    for seed in seeds_to_run:
        print(f"\n{'='*60}")
        print(f"Processing seed {seed}")
        print(f"{'='*60}")
        
        # Load existing probe results
        probe_csv_path = Path(config['tables_dir']) / f'{args.dataset}_{args.model}_K{args.K}_seed{seed}_probe.csv'
        
        if not probe_csv_path.exists():
            print(f"Warning: Probe results not found for seed {seed}: {probe_csv_path}")
            print(f"Skipping seed {seed}")
            continue
        
        df = pd.read_csv(probe_csv_path)
        print(f"\nLoaded probe results from: {probe_csv_path}")
        
        # Load per-node arrays
        arrays_path = Path(config['results_dir']) / 'arrays' / f'{args.dataset}_{args.model}_K{args.K}_seed{seed}_pernode.npz'
        
        if not arrays_path.exists():
            print(f"Warning: Per-node arrays not found for seed {seed}")
            print(f"Skipping seed {seed}")
            continue
        
        data = np.load(arrays_path)
        print(f"Loaded per-node arrays from: {arrays_path}")
        
        # Load dataset to get labels and num_classes
        from src.datasets import load_dataset as load_ds
        data_obj, num_classes, _ = load_ds(
            args.dataset,
            root_dir='data',
            planetoid_normalize=False,
            planetoid_split='public'
        )
        
        labels = data_obj.y.numpy()
        val_mask = data_obj.val_mask.numpy()
        test_mask = data_obj.test_mask.numpy()
        
        labels_val = labels[val_mask]
        labels_test = labels[test_mask]
        
        K = args.K
        
        # Compute separability metrics for each depth
        print(f"\nComputing separability metrics for k=0..{K}...")
        
        separability_metrics = []
        per_class_metrics = []
        
        for k in range(K + 1):
            H_val = data[f'H_val_{k}']
            e_val = data[f'e_val_{k}']
            H_test = data[f'H_test_{k}']
            e_test = data[f'e_test_{k}']
            
            # Load probabilities for per-class metrics
            p_val = data[f'p_val_{k}']
            
            metrics = compute_separability_metrics_at_depth(H_val, e_val, H_test, e_test)
            metrics['k'] = k
            separability_metrics.append(metrics)
            
            # Compute per-class metrics
            per_class_data = {'k': k}
            preds_val = np.argmax(p_val, axis=1)
            
            for c in range(num_classes):
                class_mask = (labels_val == c)
                n_class_total = class_mask.sum()
                
                if n_class_total > 0:
                    class_preds = preds_val[class_mask]
                    class_correct = (class_preds == c).astype(int)
                    class_entropy = H_val[class_mask]
                    
                    n_class_correct = class_correct.sum()
                    n_class_wrong = n_class_total - n_class_correct
                    
                    # Per-class accuracy
                    class_accuracy = class_correct.mean()
                    
                    # Per-class mean entropy
                    class_mean_entropy = class_entropy.mean()
                    
                    per_class_data[f'class_{c}_accuracy'] = class_accuracy
                    per_class_data[f'class_{c}_entropy'] = class_mean_entropy
                    per_class_data[f'class_{c}_n_total'] = n_class_total
                    per_class_data[f'class_{c}_n_correct'] = n_class_correct
                    per_class_data[f'class_{c}_n_wrong'] = n_class_wrong
                else:
                    per_class_data[f'class_{c}_accuracy'] = np.nan
                    per_class_data[f'class_{c}_entropy'] = np.nan
                    per_class_data[f'class_{c}_n_total'] = 0
                    per_class_data[f'class_{c}_n_correct'] = 0
                    per_class_data[f'class_{c}_n_wrong'] = 0
            
            per_class_metrics.append(per_class_data)
        
        # Merge with original probe results
        sep_df = pd.DataFrame(separability_metrics)
        df_enriched = df.merge(sep_df, on='k')
        
        # Perform constrained selection
        print(f"\nPerforming constrained depth selection (eps_acc={args.eps_acc})...")
        k_star_sep, k_star_method, k_best_val_acc = select_k_star_constrained(df_enriched, args.eps_acc)
        
        # Compute Spearman correlations
        print(f"\nComputing Spearman correlations...")
        correlations = compute_spearman_correlations(df_enriched)
        
        for key, val in correlations.items():
            if 'rho' in key:
                print(f"  {key}: {val:.4f}")
        
        # Compute baseline selection methods for comparison
        print(f"\nComputing baseline selection methods for comparison...")
        k_star_nll = select_k_star_nll(df_enriched)
        k_star_combined = select_k_star_combined(df_enriched, lambda_val=0.1)
        k_star_top3 = select_k_star_top3(df_enriched)
        
        print(f"  k_star_nll (argmin val_nll) = {k_star_nll}")
        print(f"  k_star_combined (argmin val_nll + 0.1*entropy) = {k_star_combined}")
        print(f"  k_star_top3 (top-3 acc, max AUROC) = {k_star_top3}")
        print(f"  k_star_sep (constrained AUROC) = {k_star_sep}")
        
        # Save enriched CSV
        output_csv = Path(config['tables_dir']) / f'{args.dataset}_{args.model}_K{args.K}_seed{seed}_probe_with_separability.csv'
        df_enriched.to_csv(output_csv, index=False)
        print(f"\n[DONE] Enriched results saved to: {output_csv}")
        
        # Save summary CSV
        summary = {
            'dataset': args.dataset,
            'model': args.model,
            'K': args.K,
            'seed': seed,
            'k_best_val_acc': k_best_val_acc,
            'k_star_nll': k_star_nll,
            'k_star_combined': k_star_combined,
            'k_star_top3': k_star_top3,
            'k_star_sep': k_star_sep,
            'k_star_method': k_star_method,
            'eps_acc': args.eps_acc,
        }
        summary.update(correlations)
        
        summary_df = pd.DataFrame([summary])
        summary_csv = Path(config['tables_dir']) / f'{args.dataset}_{args.model}_K{args.K}_seed{seed}_separability_summary.csv'
        summary_df.to_csv(summary_csv, index=False)
        print(f"[DONE] Summary saved to: {summary_csv}")
        
        # Generate individual seed plot
        print(f"\nGenerating plots...")
        
        # Create hierarchical directory structure
        plot_dir = Path(config['figures_dir']) / args.dataset / args.model / f'K_{args.K}'
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert per-class metrics to DataFrame
        df_per_class = pd.DataFrame(per_class_metrics)
        
        plot_path = plot_dir / f'{args.dataset}_{args.model}_k{args.K}_seed{seed}_separability_vs_k.png'
        plot_separability_vs_depth(df_enriched, df_per_class, num_classes, summary, plot_path)
        
        print(f"\n{'='*60}")
        print(f"[DONE] Separability analysis complete for seed {seed}!")
        print(f"  k_best_val_acc = {k_best_val_acc}")
        print(f"  k_star_sep = {k_star_sep} (method: {k_star_method})")
        print(f"{'='*60}\n")
    
    # If running all seeds, generate aggregated plot
    if run_all_seeds and len(seeds_to_run) > 1:
        print(f"\n{'='*60}")
        print(f"Generating aggregated plot across all seeds")
        print(f"{'='*60}")
        plot_aggregated_seeds(args.dataset, args.model, args.K, seeds_to_run, config)
        print(f"\n{'='*60}")
        print(f"[DONE] All seeds complete with aggregated plot!")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
