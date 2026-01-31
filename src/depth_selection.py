"""Depth selection via validation NLL and optional entropy-based criteria."""

import argparse
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def select_optimal_depth(probe_results_df, selection_method='nll', lambda_val=0.0):
    """
    Select optimal depth from probe results.
    
    Args:
        probe_results_df: DataFrame with columns [k, val_nll, val_entropy_mean, ...]
        selection_method: 'nll' or 'combined'
        lambda_val: Weight for entropy term in combined selection
        
    Returns:
        k_star: Selected optimal depth
        selection_score: The score used for selection at k_star
    """
    if selection_method == 'nll':
        # Primary selection: argmin_k val_nll(k)
        k_star = probe_results_df.loc[probe_results_df['val_nll'].idxmin(), 'k']
        score = probe_results_df.loc[probe_results_df['k'] == k_star, 'val_nll'].values[0]
        
    elif selection_method == 'combined':
        # Combined selection: argmin_k (val_nll(k) + λ * val_entropy_mean(k))
        probe_results_df['combined_score'] = (
            probe_results_df['val_nll'] + lambda_val * probe_results_df['val_entropy_mean']
        )
        k_star = probe_results_df.loc[probe_results_df['combined_score'].idxmin(), 'k']
        score = probe_results_df.loc[probe_results_df['k'] == k_star, 'combined_score'].values[0]
        
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
    
    return int(k_star), float(score)


def depth_selection_for_run(dataset_name, model_name, seed, config, selection_method='nll', lambda_val=0.0):
    """
    Perform depth selection for a single run.
    
    Args:
        dataset_name: Dataset name
        model_name: Model name
        seed: Random seed
        config: Configuration dict
        selection_method: 'nll' or 'combined'
        lambda_val: Lambda value for combined selection
        
    Returns:
        dict with selected depth and corresponding metrics
    """
    # Load probe results
    probe_file = Path(config['tables_dir']) / f'{dataset_name}_{model_name}_seed{seed}_probe.csv'
    
    if not probe_file.exists():
        print(f"⚠ Probe results not found: {probe_file}")
        return None
    
    probe_df = pd.read_csv(probe_file)
    
    # Select optimal depth
    k_star, score = select_optimal_depth(probe_df, selection_method, lambda_val)
    
    # Get metrics at selected depth
    selected_row = probe_df[probe_df['k'] == k_star].iloc[0]
    
    # Get metrics at K_max for comparison
    K_max = probe_df['k'].max()
    kmax_row = probe_df[probe_df['k'] == K_max].iloc[0]
    
    result = {
        'dataset': dataset_name,
        'model': model_name,
        'seed': seed,
        'selection_method': selection_method,
        'lambda': lambda_val,
        'k_star': k_star,
        'K_max': K_max,
        # Performance at k*
        'k_star_val_nll': selected_row['val_nll'],
        'k_star_val_acc': selected_row['val_acc'],
        'k_star_val_entropy': selected_row['val_entropy_mean'],
        'k_star_test_acc': selected_row['test_acc'],
        'k_star_test_entropy': selected_row['test_entropy_mean'],
        # Performance at K_max
        'K_max_val_nll': kmax_row['val_nll'],
        'K_max_val_acc': kmax_row['val_acc'],
        'K_max_val_entropy': kmax_row['val_entropy_mean'],
        'K_max_test_acc': kmax_row['test_acc'],
        'K_max_test_entropy': kmax_row['test_entropy_mean'],
    }
    
    return result


def summarize_across_seeds(dataset_name, model_name, config, selection_method='nll', lambda_val=0.0):
    """
    Summarize depth selection results across all seeds.
    
    Args:
        dataset_name: Dataset name
        model_name: Model name
        config: Configuration dict
        selection_method: 'nll' or 'combined'
        lambda_val: Lambda value for combined selection
        
    Returns:
        DataFrame with mean ± std across seeds
    """
    results = []
    
    for seed in config['seeds']:
        result = depth_selection_for_run(
            dataset_name, model_name, seed, config, 
            selection_method=selection_method, lambda_val=lambda_val
        )
        if result is not None:
            results.append(result)
    
    if len(results) == 0:
        print(f"⚠ No results found for {dataset_name} {model_name}")
        return None
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Compute summary statistics
    summary = {
        'dataset': dataset_name,
        'model': model_name,
        'selection_method': selection_method,
        'lambda': lambda_val,
        'n_seeds': len(results),
    }
    
    # Mean ± std for k*
    summary['k_star_mean'] = results_df['k_star'].mean()
    summary['k_star_std'] = results_df['k_star'].std()
    
    # Test accuracy at k*
    summary['k_star_test_acc_mean'] = results_df['k_star_test_acc'].mean()
    summary['k_star_test_acc_std'] = results_df['k_star_test_acc'].std()
    
    # Test entropy at k*
    summary['k_star_test_entropy_mean'] = results_df['k_star_test_entropy'].mean()
    summary['k_star_test_entropy_std'] = results_df['k_star_test_entropy'].std()
    
    # Test accuracy at K_max
    summary['K_max_test_acc_mean'] = results_df['K_max_test_acc'].mean()
    summary['K_max_test_acc_std'] = results_df['K_max_test_acc'].std()
    
    # Test entropy at K_max
    summary['K_max_test_entropy_mean'] = results_df['K_max_test_entropy'].mean()
    summary['K_max_test_entropy_std'] = results_df['K_max_test_entropy'].std()
    
    # Compute improvement
    summary['test_acc_improvement'] = (
        summary['k_star_test_acc_mean'] - summary['K_max_test_acc_mean']
    )
    summary['test_entropy_reduction'] = (
        summary['K_max_test_entropy_mean'] - summary['k_star_test_entropy_mean']
    )
    
    return pd.DataFrame([summary]), results_df


def main():
    parser = argparse.ArgumentParser(description='Depth selection via validation NLL')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--method', type=str, default='nll',
                       choices=['nll', 'combined'],
                       help='Selection method')
    parser.add_argument('--lambda_val', type=float, default=0.0,
                       help='Lambda value for combined selection')
    
    args = parser.parse_args()
    
    # Convert config module to dict
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    # Create output directory
    Path(config['tables_dir']).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Depth Selection: {args.dataset} {args.model}")
    print(f"Method: {args.method}, λ={args.lambda_val}")
    print(f"{'='*60}\n")
    
    # Summarize across seeds
    summary_df, detailed_df = summarize_across_seeds(
        args.dataset, args.model, config,
        selection_method=args.method, lambda_val=args.lambda_val
    )
    
    if summary_df is not None:
        # Save summary
        method_suffix = f"_{args.method}" if args.method != 'nll' else ''
        lambda_suffix = f"_lambda{args.lambda_val}" if args.method == 'combined' else ''
        
        summary_file = (
            Path(config['tables_dir']) / 
            f'summary_{args.dataset}_{args.model}{method_suffix}{lambda_suffix}.csv'
        )
        summary_df.to_csv(summary_file, index=False)
        
        # Save detailed results
        detailed_file = (
            Path(config['tables_dir']) / 
            f'detailed_{args.dataset}_{args.model}{method_suffix}{lambda_suffix}.csv'
        )
        detailed_df.to_csv(detailed_file, index=False)
        
        # Print results
        print("\n📊 Summary Across Seeds:")
        print(f"  Selected depth k*: {summary_df['k_star_mean'].values[0]:.2f} ± {summary_df['k_star_std'].values[0]:.2f}")
        print(f"\n  Test Accuracy:")
        print(f"    At k*:    {summary_df['k_star_test_acc_mean'].values[0]:.4f} ± {summary_df['k_star_test_acc_std'].values[0]:.4f}")
        print(f"    At K_max: {summary_df['K_max_test_acc_mean'].values[0]:.4f} ± {summary_df['K_max_test_acc_std'].values[0]:.4f}")
        print(f"    Improvement: {summary_df['test_acc_improvement'].values[0]:.4f}")
        
        print(f"\n  Test Entropy:")
        print(f"    At k*:    {summary_df['k_star_test_entropy_mean'].values[0]:.4f} ± {summary_df['k_star_test_entropy_std'].values[0]:.4f}")
        print(f"    At K_max: {summary_df['K_max_test_entropy_mean'].values[0]:.4f} ± {summary_df['K_max_test_entropy_std'].values[0]:.4f}")
        print(f"    Reduction: {summary_df['test_entropy_reduction'].values[0]:.4f}")
        
        print(f"\n✓ Results saved:")
        print(f"  Summary: {summary_file}")
        print(f"  Detailed: {detailed_file}")


if __name__ == '__main__':
    main()
