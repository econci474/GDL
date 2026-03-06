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
            
            label = f'C{c} (n={n_total}: {n_correct}[OK]/{n_wrong}[FAIL])'
            ax.plot(df_per_class['k'], df_per_class[entropy_col],
                   'o-', linewidth=1.5, markersize=5, label=label, color=colors[c])
    
    ax.set_xlabel('Depth k', fontsize=11, fontweight='bold')
    ax.set_ylabel('Per-Class Mean Entropy', fontsize=11, fontweight='bold')
    ax.set_title('Per-Class Mean Entropy by Depth', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=1, loc='best')
    
    # Update legend on first subplot
    axes[0, 0].legend(loc='best', fontsize=8)

    dataset = summary.get('dataset', '')
    model   = summary.get('model', '')
    seed    = summary.get('seed', '')
    fig.suptitle(
        f"Linear Probe Analysis: {model}, {dataset}, Seed {seed}",
        fontsize=13, fontweight='bold', y=1.01
    )

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
        
        # (k* depth selection lines removed)
    
    seeds_str = ", ".join(str(s) for s in seeds)
    fig.suptitle(
        f"Linear Probe Analysis: {model}, {dataset} -- Mean +/- Std across Seeds ({seeds_str})",
        fontsize=13, fontweight='bold', y=1.01
    )

    plt.tight_layout()
    
    # Save to hierarchical directory
    plot_dir = Path(config['figures_dir']) / dataset / model / f'K_{K}'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = plot_dir / f'{dataset}_{model}_k{K}_seed_all_separability_vs_k.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Aggregated plot saved to: {output_path}")
    print(f"  Aggregated across {len(seeds)} seeds")


def plot_aggregated_seeds_per_class(dataset, model, K, seeds, config):
    """
    6-panel (3x2) aggregated plot across seeds including per-class breakdown.

    Top 4 panels: AUROC, Cohen's d, Validation Accuracy, Mean Entropy (mean±std).
    Bottom 2 panels: Per-Class Accuracy, Per-Class Mean Entropy (mean±std per class).

    Saved as: {dataset}_{model}_k{K}_seed_all_separability_vs_k_per_class.png
    """
    # ---------- Load enriched (aggregate) data ----------
    all_dfs = []
    for seed in seeds:
        csv_path = Path(config['tables_dir']) / f'{dataset}_{model}_K{K}_seed{seed}_probe_with_separability.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['seed'] = seed
            all_dfs.append(df)
        else:
            print(f"Warning: Missing enriched data for seed {seed}, skipping")

    if len(all_dfs) == 0:
        print("No enriched data found for aggregation")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    metrics = ['val_auroc_err_from_entropy', 'val_cohens_d', 'val_acc',
               'val_entropy_auc_correct', 'val_entropy_auc_incorrect']
    agg_df = combined_df.groupby('k')[metrics].agg(['mean', 'std']).reset_index()
    k_values = agg_df['k'].values

    # ---------- Load per-class data ----------
    all_pc_dfs = []
    for seed in seeds:
        pc_path = Path(config['tables_dir']) / f'{dataset}_{model}_K{K}_seed{seed}_per_class.csv'
        if pc_path.exists():
            df_pc = pd.read_csv(pc_path)
            df_pc['seed'] = seed
            all_pc_dfs.append(df_pc)
        else:
            print(f"Warning: Missing per-class data for seed {seed}, skipping per-class panels")

    # Detect number of classes from columns
    if len(all_pc_dfs) > 0:
        sample_cols = all_pc_dfs[0].columns.tolist()
        num_classes = sum(1 for c in sample_cols if c.startswith('class_') and c.endswith('_accuracy'))
        combined_pc = pd.concat(all_pc_dfs, ignore_index=True)
        # Aggregate per-class accuracy and entropy
        pc_acc_cols     = [f'class_{c}_accuracy' for c in range(num_classes)]
        pc_ent_cols     = [f'class_{c}_entropy'  for c in range(num_classes)]
        pc_n_total_cols = [f'class_{c}_n_total'  for c in range(num_classes)]
        pc_agg = combined_pc.groupby('k')[pc_acc_cols + pc_ent_cols].agg(['mean', 'std']).reset_index()
        # n_total is fixed per class — take from first seed
        n_total_by_class = {
            c: int(all_pc_dfs[0][f'class_{c}_n_total'].iloc[0])
            for c in range(num_classes)
        }
        has_per_class = True
    else:
        has_per_class = False
        num_classes = 0

    # ---------- Build figure ----------
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    colors = plt.cm.tab10(np.arange(max(num_classes, 10)))

    # Panel 1: AUROC
    ax = axes[0, 0]
    mean = agg_df[('val_auroc_err_from_entropy', 'mean')].values
    std  = agg_df[('val_auroc_err_from_entropy', 'std')].values
    ax.plot(k_values, mean, 'o-', label=f'Mean (n={len(seeds)} seeds)', color='tab:blue', linewidth=2)
    ax.fill_between(k_values, mean - std, mean + std, alpha=0.2, color='tab:blue')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlabel('Depth k', fontsize=11, fontweight='bold')
    ax.set_ylabel('AUROC', fontsize=11, fontweight='bold')
    ax.set_title('Error Detection AUROC vs Depth', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 2: Cohen's d
    ax = axes[0, 1]
    mean = agg_df[('val_cohens_d', 'mean')].values
    std  = agg_df[('val_cohens_d', 'std')].values
    ax.plot(k_values, mean, 'o-', label=f'Mean (n={len(seeds)} seeds)', color='tab:orange', linewidth=2)
    ax.fill_between(k_values, mean - std, mean + std, alpha=0.2, color='tab:orange')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Depth k', fontsize=11, fontweight='bold')
    ax.set_ylabel("Cohen's d", fontsize=11, fontweight='bold')
    ax.set_title("Entropy Separability (Cohen's d) vs Depth", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 3: Validation Accuracy
    ax = axes[1, 0]
    mean = agg_df[('val_acc', 'mean')].values
    std  = agg_df[('val_acc', 'std')].values
    ax.plot(k_values, mean, 'o-', label=f'Mean (n={len(seeds)} seeds)', color='tab:green', linewidth=2)
    ax.fill_between(k_values, mean - std, mean + std, alpha=0.2, color='tab:green')
    ax.set_xlabel('Depth k', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Validation Accuracy vs Depth', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 4: Mean Entropy correct vs incorrect
    ax = axes[1, 1]
    mc  = agg_df[('val_entropy_auc_correct',   'mean')].values
    sc  = agg_df[('val_entropy_auc_correct',   'std')].values
    mi  = agg_df[('val_entropy_auc_incorrect', 'mean')].values
    si  = agg_df[('val_entropy_auc_incorrect', 'std')].values
    ax.plot(k_values, mc, 'o-', label='Correct Predictions',   color='tab:blue', linewidth=2)
    ax.fill_between(k_values, mc - sc, mc + sc, alpha=0.2, color='tab:blue')
    ax.plot(k_values, mi, 's-', label='Incorrect Predictions', color='tab:red',  linewidth=2)
    ax.fill_between(k_values, mi - si, mi + si, alpha=0.2, color='tab:red')
    ax.set_xlabel('Depth k', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Entropy', fontsize=11, fontweight='bold')
    ax.set_title('Mean Entropy: Correct vs Incorrect', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 5 & 6: Per-class (if data available)
    if has_per_class:
        # Sort classes by descending n_total for legend
        sorted_classes = sorted(range(num_classes), key=lambda c: n_total_by_class[c])

        # Panel 5: Per-class Accuracy
        ax = axes[2, 0]
        for c in sorted_classes:
            m = pc_agg[(f'class_{c}_accuracy', 'mean')].values
            s = pc_agg[(f'class_{c}_accuracy', 'std')].values
            n = n_total_by_class[c]
            ax.plot(k_values, m, 'o-', linewidth=1.5, markersize=5,
                    label=f'C{c} (n={n})', color=colors[c])
            ax.fill_between(k_values, m - s, m + s, alpha=0.12, color=colors[c])
        ax.set_xlabel('Depth k', fontsize=11, fontweight='bold')
        ax.set_ylabel('Per-Class Accuracy', fontsize=11, fontweight='bold')
        ax.set_title('Per-Class Validation Accuracy by Depth', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

        # Panel 6: Per-class Entropy
        ax = axes[2, 1]
        for c in sorted_classes:
            m = pc_agg[(f'class_{c}_entropy', 'mean')].values
            s = pc_agg[(f'class_{c}_entropy', 'std')].values
            n = n_total_by_class[c]
            ax.plot(k_values, m, 'o-', linewidth=1.5, markersize=5,
                    label=f'C{c} (n={n})', color=colors[c])
            ax.fill_between(k_values, m - s, m + s, alpha=0.12, color=colors[c])
        ax.set_xlabel('Depth k', fontsize=11, fontweight='bold')
        ax.set_ylabel('Per-Class Mean Entropy', fontsize=11, fontweight='bold')
        ax.set_title('Per-Class Mean Entropy by Depth', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, ncol=1, loc='best')
    else:
        axes[2, 0].set_visible(False)
        axes[2, 1].set_visible(False)

    # ---------- Title & save ----------
    seeds_str = ", ".join(str(s) for s in seeds)
    fig.suptitle(
        f"Linear Probe Analysis: {model}, {dataset} -- Mean +/- Std across Seeds ({seeds_str})",
        fontsize=13, fontweight='bold', y=1.01
    )

    plt.tight_layout()

    plot_dir = Path(config['figures_dir']) / dataset / model / f'K_{K}'
    plot_dir.mkdir(parents=True, exist_ok=True)

    output_path = plot_dir / f'{dataset}_{model}_k{K}_seed_all_separability_vs_k_per_class.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Per-class aggregated plot saved to: {output_path}")


def main():

    parser = argparse.ArgumentParser(description='Compute separability metrics and constrained depth selection')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--seed', type=str, default='0',
                       help='Seed value or "all" to run all seeds from config')
    parser.add_argument('--split', type=str, default=None,
                       help='Split index for heterophilous datasets, or "all" to average across '
                            'all 10 splits. Omit for homophilous datasets (Cora, PubMed).')
    parser.add_argument('--eps_acc', type=float, default=0.02,
                       help='Accuracy tolerance for constrained selection (default: 0.02)')

    args = parser.parse_args()

    # Convert config module to dict
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}

    # ---------- Parse --split ----------
    # splits_to_run: list of ints, or [None] for homophilous (no suffix)
    if args.split is None:
        splits_to_run = [None]
        split_label   = ''              # no suffix in file names
    elif args.split.lower() == 'all':
        splits_to_run = list(range(10))
        split_label   = '_all_splits'
    else:
        splits_to_run = [int(args.split)]
        split_label   = f'_split{args.split}'

    # ---------- Parse --seed ----------
    if args.seed.lower() == 'all':
        seeds_to_run  = config['seeds']
        run_all_seeds = True
    else:
        seeds_to_run  = [int(args.seed)]
        run_all_seeds = False

    seed_label = '_all_seeds' if run_all_seeds else f'_seed{seeds_to_run[0]}'

    print(f"\n{'='*60}")
    print(f"Separability Metrics: {args.model} on {args.dataset}")
    print(f"K={args.K}, seeds={seeds_to_run}, splits={splits_to_run}, eps_acc={args.eps_acc}")
    print(f"{'='*60}")

    # ---------- Load dataset once (for labels/masks) ----------
    from src.datasets import load_dataset as load_ds
    data_obj, num_classes, _ = load_ds(
        args.dataset,
        root_dir='data',
        planetoid_normalize=False,
        planetoid_split='public'
    )
    labels    = data_obj.y.numpy()
    # val_mask may be 1-D (homophilous) or 2-D (N×10, heterophilous)
    val_mask_raw  = data_obj.val_mask
    test_mask_raw = data_obj.test_mask

    # ---------- Helper: get (probe_csv, arrays_npz, val_mask, test_mask) for a (seed, split) ----------
    def _get_paths_and_masks(seed, split):
        sfx = '' if split is None else f'_split{split}'
        probe_path  = Path(config['tables_dir']) / f'{args.dataset}_{args.model}_K{args.K}_seed{seed}{sfx}_probe.csv'
        arrays_path = Path(config['results_dir']) / 'arrays' / f'{args.dataset}_{args.model}_K{args.K}_seed{seed}{sfx}_pernode.npz'
        # Select correct mask slice
        if val_mask_raw.dim() == 2:
            s = split if split is not None else 0
            vm = val_mask_raw[:, s].numpy()
            tm = test_mask_raw[:, s].numpy()
        else:
            vm = val_mask_raw.numpy()
            tm = test_mask_raw.numpy()
        return probe_path, arrays_path, vm, tm

    # ---------- Collect all (seed, split) combinations ----------
    combinations = [(s, sp) for s in seeds_to_run for sp in splits_to_run]
    print(f"Total (seed, split) combinations: {len(combinations)}")

    # Aggregation tag for file names
    agg_tag = seed_label + split_label   # e.g. '_seed0_all_splits' or '_all_seeds_split0'

    # We will collect enriched DataFrames and per-class DataFrames for the aggregated plot
    all_enriched_dfs  = []
    all_per_class_dfs = []

    for seed, split in combinations:
        sfx = '' if split is None else f'_split{split}'
        combo_label = f"seed {seed}" + ('' if split is None else f", split {split}")
        print(f"\n{'='*60}")
        print(f"Processing {combo_label}")
        print(f"{'='*60}")

        probe_csv_path, arrays_path, val_mask, test_mask = _get_paths_and_masks(seed, split)

        if not probe_csv_path.exists():
            print(f"Warning: Probe CSV not found: {probe_csv_path} -- skipping")
            continue
        if not arrays_path.exists():
            print(f"Warning: Arrays npz not found: {arrays_path} -- skipping")
            continue

        df   = pd.read_csv(probe_csv_path)
        data = np.load(arrays_path)
        print(f"Loaded: {probe_csv_path.name}")

        labels_val  = labels[val_mask]
        K = args.K

        # ---------- Compute metrics for each depth ----------
        separability_metrics = []
        per_class_metrics    = []

        for k in range(K + 1):
            H_val  = data[f'H_val_{k}']
            e_val  = data[f'e_val_{k}']
            H_test = data[f'H_test_{k}']
            e_test = data[f'e_test_{k}']
            p_val  = data[f'p_val_{k}']

            metrics      = compute_separability_metrics_at_depth(H_val, e_val, H_test, e_test)
            metrics['k'] = k
            separability_metrics.append(metrics)

            per_class_data = {'k': k}
            preds_val      = np.argmax(p_val, axis=1)

            for c in range(num_classes):
                class_mask    = (labels_val == c)
                n_class_total = class_mask.sum()
                if n_class_total > 0:
                    class_preds   = preds_val[class_mask]
                    class_correct = (class_preds == c).astype(int)
                    class_entropy = H_val[class_mask]
                    n_correct     = class_correct.sum()
                    per_class_data[f'class_{c}_accuracy'] = class_correct.mean()
                    per_class_data[f'class_{c}_entropy']  = class_entropy.mean()
                    per_class_data[f'class_{c}_n_total']  = n_class_total
                    per_class_data[f'class_{c}_n_correct']= n_correct
                    per_class_data[f'class_{c}_n_wrong']  = n_class_total - n_correct
                else:
                    for col in ['accuracy', 'entropy']:
                        per_class_data[f'class_{c}_{col}'] = np.nan
                    for col in ['n_total', 'n_correct', 'n_wrong']:
                        per_class_data[f'class_{c}_{col}'] = 0
            per_class_metrics.append(per_class_data)

        sep_df     = pd.DataFrame(separability_metrics)
        df_enriched = df.merge(sep_df, on='k')
        df_enriched['seed']  = seed
        df_enriched['split'] = split if split is not None else -1
        all_enriched_dfs.append(df_enriched)

        df_per_class = pd.DataFrame(per_class_metrics)
        df_per_class['seed']  = seed
        df_per_class['split'] = split if split is not None else -1
        all_per_class_dfs.append(df_per_class)

        # ---------- Constrained selection & correlations ----------
        print(f"\nPerforming constrained depth selection (eps_acc={args.eps_acc})...")
        k_star_sep, k_star_method, k_best_val_acc = select_k_star_constrained(df_enriched, args.eps_acc)
        correlations = compute_spearman_correlations(df_enriched)
        for key, val_v in correlations.items():
            if 'rho' in key:
                print(f"  {key}: {val_v:.4f}")
        k_star_nll      = select_k_star_nll(df_enriched)
        k_star_combined = select_k_star_combined(df_enriched, lambda_val=0.1)
        k_star_top3     = select_k_star_top3(df_enriched)
        print(f"  k_star_nll={k_star_nll}  k_star_combined={k_star_combined}  "
              f"k_star_top3={k_star_top3}  k_star_sep={k_star_sep}")

        # ---------- Save per-seed/split outputs ----------
        out_sfx = f'_seed{seed}{sfx}'
        output_csv   = Path(config['tables_dir']) / f'{args.dataset}_{args.model}_K{args.K}{out_sfx}_probe_with_separability.csv'
        summary_csv  = Path(config['tables_dir']) / f'{args.dataset}_{args.model}_K{args.K}{out_sfx}_separability_summary.csv'
        per_class_csv= Path(config['tables_dir']) / f'{args.dataset}_{args.model}_K{args.K}{out_sfx}_per_class.csv'

        df_enriched.to_csv(output_csv, index=False)
        df_per_class.to_csv(per_class_csv, index=False)

        summary = {
            'dataset': args.dataset, 'model': args.model, 'K': args.K,
            'seed': seed, 'split': split,
            'k_best_val_acc': k_best_val_acc,
            'k_star_nll': k_star_nll, 'k_star_combined': k_star_combined,
            'k_star_top3': k_star_top3, 'k_star_sep': k_star_sep,
            'k_star_method': k_star_method, 'eps_acc': args.eps_acc,
        }
        summary.update(correlations)
        pd.DataFrame([summary]).to_csv(summary_csv, index=False)
        print(f"[DONE] Outputs saved with suffix '{out_sfx}'")

        # Individual plot only when there's a single (seed, split) or we're iterating single-seed
        if len(combinations) == 1 or (run_all_seeds and len(splits_to_run) == 1):
            plot_dir = Path(config['figures_dir']) / args.dataset / args.model / f'K_{args.K}'
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / f'{args.dataset}_{args.model}_k{args.K}{out_sfx}_separability_vs_k.png'
            summary['seed'] = seed   # ensure suptitle uses correct seed
            plot_separability_vs_depth(df_enriched, df_per_class, num_classes, summary, plot_path)
            print(f"  Individual plot saved: {plot_path.name}")

        print(f"\n{'='*60}")
        print(f"[DONE] {combo_label}: k_best_val_acc={k_best_val_acc}  k_star_sep={k_star_sep}")
        print(f"{'='*60}\n")

    # ---------- Aggregated plot across all collected combinations ----------
    if len(all_enriched_dfs) > 1:
        print(f"\n{'='*60}")
        print(f"Generating aggregated plots ({len(all_enriched_dfs)} combinations: {agg_tag})")
        print(f"{'='*60}")

        # Build a temporary config pointing to the combined CSVs that the aggregated
        # plot functions can find by their standard naming convention.
        # Simpler: pass DataFrames directly via a local helper.

        combined_enriched  = pd.concat(all_enriched_dfs,  ignore_index=True)
        combined_per_class = pd.concat(all_per_class_dfs, ignore_index=True)

        metrics_cols = ['val_auroc_err_from_entropy', 'val_cohens_d', 'val_acc',
                        'val_entropy_auc_correct', 'val_entropy_auc_incorrect']
        agg_df   = combined_enriched.groupby('k')[metrics_cols].agg(['mean', 'std']).reset_index()
        k_values = agg_df['k'].values

        pc_acc_cols = [c for c in combined_per_class.columns if c.endswith('_accuracy') and c.startswith('class_')]
        pc_ent_cols = [c for c in combined_per_class.columns if c.endswith('_entropy')  and c.startswith('class_')]
        pc_agg = combined_per_class.groupby('k')[pc_acc_cols + pc_ent_cols].agg(['mean', 'std']).reset_index()
        n_total_by_class = {
            int(c.split('_')[1]): int(all_per_class_dfs[0][c.replace('_accuracy','_n_total')].iloc[0])
            for c in pc_acc_cols
        }
        sorted_classes = sorted(n_total_by_class.keys(), key=lambda c: n_total_by_class[c])

        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        colors = plt.cm.tab10(np.arange(max(num_classes, 10)))

        def _plot_mean_std(ax, kv, m, s, color, label, marker='o'):
            ax.plot(kv, m, f'{marker}-', label=label, color=color, linewidth=2)
            ax.fill_between(kv, m - s, m + s, alpha=0.2, color=color)

        n_combos = len(all_enriched_dfs)
        lbl = f'Mean (n={n_combos})'

        ax = axes[0, 0]
        _plot_mean_std(ax, k_values, agg_df[('val_auroc_err_from_entropy','mean')].values,
                       agg_df[('val_auroc_err_from_entropy','std')].values, 'tab:blue', lbl)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
        ax.set(xlabel='Depth k', ylabel='AUROC', title='Error Detection AUROC vs Depth', ylim=[0,1])
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

        ax = axes[0, 1]
        _plot_mean_std(ax, k_values, agg_df[('val_cohens_d','mean')].values,
                       agg_df[('val_cohens_d','std')].values, 'tab:orange', lbl)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set(xlabel='Depth k', ylabel="Cohen's d", title="Entropy Separability (Cohen's d) vs Depth")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

        ax = axes[1, 0]
        _plot_mean_std(ax, k_values, agg_df[('val_acc','mean')].values,
                       agg_df[('val_acc','std')].values, 'tab:green', lbl)
        ax.set(xlabel='Depth k', ylabel='Accuracy', title='Validation Accuracy vs Depth', ylim=[0,1])
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

        ax = axes[1, 1]
        _plot_mean_std(ax, k_values, agg_df[('val_entropy_auc_correct','mean')].values,
                       agg_df[('val_entropy_auc_correct','std')].values, 'tab:blue', 'Correct', 'o')
        _plot_mean_std(ax, k_values, agg_df[('val_entropy_auc_incorrect','mean')].values,
                       agg_df[('val_entropy_auc_incorrect','std')].values, 'tab:red', 'Incorrect', 's')
        ax.set(xlabel='Depth k', ylabel='Mean Entropy', title='Mean Entropy: Correct vs Incorrect')
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

        ax = axes[2, 0]
        for c in sorted_classes:
            col = f'class_{c}_accuracy'
            m = pc_agg[(col,'mean')].values; s = pc_agg[(col,'std')].values
            ax.plot(k_values, m, 'o-', linewidth=1.5, markersize=5,
                    label=f'C{c} (n={n_total_by_class[c]})', color=colors[c])
            ax.fill_between(k_values, m-s, m+s, alpha=0.12, color=colors[c])
        ax.set(xlabel='Depth k', ylabel='Per-Class Accuracy',
               title='Per-Class Validation Accuracy by Depth', ylim=[0,1.05])
        ax.grid(True, alpha=0.3)

        ax = axes[2, 1]
        for c in sorted_classes:
            col = f'class_{c}_entropy'
            m = pc_agg[(col,'mean')].values; s = pc_agg[(col,'std')].values
            ax.plot(k_values, m, 'o-', linewidth=1.5, markersize=5,
                    label=f'C{c} (n={n_total_by_class[c]})', color=colors[c])
            ax.fill_between(k_values, m-s, m+s, alpha=0.12, color=colors[c])
        ax.set(xlabel='Depth k', ylabel='Per-Class Mean Entropy',
               title='Per-Class Mean Entropy by Depth')
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9, ncol=1, loc='best')

        # Title reflecting what was averaged
        if run_all_seeds and len(splits_to_run) > 1:
            avg_desc = f"Mean +/- Std across Seeds {seeds_to_run} x Splits 0..{len(splits_to_run)-1}"
        elif run_all_seeds:
            sp_str = '' if splits_to_run[0] is None else f', Split {splits_to_run[0]}'
            avg_desc = f"Mean +/- Std across Seeds {seeds_to_run}{sp_str}"
        else:
            avg_desc = f"Mean +/- Std across Splits 0..{len(splits_to_run)-1}, Seed {seeds_to_run[0]}"

        fig.suptitle(f"Linear Probe Analysis: {args.model}, {args.dataset} -- {avg_desc}",
                     fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()

        plot_dir = Path(config['figures_dir']) / args.dataset / args.model / f'K_{args.K}'
        plot_dir.mkdir(parents=True, exist_ok=True)
        out_plot = plot_dir / f'{args.dataset}_{args.model}_k{args.K}{agg_tag}_separability_vs_k_per_class.png'
        plt.savefig(out_plot, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Aggregated per-class plot saved: {out_plot.name}")

        print(f"\n{'='*60}")
        print(f"[DONE] Aggregated plot complete!")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
