"""Generate comparison tables for heterophilous datasets with per-split results."""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def load_split_results(dataset_name, model_name, K, seed, split_idx, runs_dir):
    """Load results from a specific split."""
    log_file = Path(runs_dir) / dataset_name / model_name / f'seed_{seed}' / f'K_{K}' / f'split_{split_idx}' / 'train_log.csv'
    
    if not log_file.exists():
        return None
    
    df = pd.read_csv(log_file)
    best_epoch = df['val_loss'].idxmin()
    
    return {
        'dataset': dataset_name,
        'model': model_name,
        'K': K,
        'seed': seed,
        'split': split_idx,
        'best_epoch': best_epoch + 1,
        'train_acc': df.loc[best_epoch, 'train_acc'],
        'val_acc': df.loc[best_epoch, 'val_acc'],
        'test_acc': df.loc[best_epoch, 'test_acc'],
        'val_loss': df.loc[best_epoch, 'val_loss']
    }


def generate_heterophilous_table(datasets, model, K_values, seeds, split_idx, runs_dir):
    """Generate comparison table for heterophilous datasets at a specific split."""
    results = []
    
    for dataset in datasets:
        for K in K_values:
            for seed in seeds:
                result = load_split_results(dataset, model, K, seed, split_idx, runs_dir)
                if result:
                    results.append(result)
    
    if not results:
        return None
    
    df = pd.DataFrame(results)
    
    # Pivot to show results nicely
    summary = df.pivot_table(
        index=['dataset', 'K'],
        values=['test_acc', 'val_acc'],
        aggfunc={
            'test_acc': ['mean', 'std'],
            'val_acc': ['mean', 'std']
        }
    ).round(4)
    
    return df, summary


def main():
    parser = argparse.ArgumentParser(description='Generate comparison tables for heterophilous datasets')
    parser.add_argument('--model', type=str, default='GCN',
                       help='Model name (default: GCN)')
    parser.add_argument('--split', type=int, default=0,
                       help='Split index to analyze (default: 0)')
    parser.add_argument('--K-values', type=int, nargs='+', default=[3, 8],
                       help='K values to include (default: 3 8)')
    
    args = parser.parse_args()
    
    # Load config
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    # Heterophilous datasets
    heterophilous_datasets = ['Roman-empire', 'Minesweeper']
    
    print(f"\n{'='*70}")
    print(f"Heterophilous Dataset Comparison Table")
    print(f"Model: {args.model}, Split: {args.split}, K values: {args.K_values}")
    print(f"{'='*70}\n")
    
    # Generate table
    df, summary = generate_heterophilous_table(
        heterophilous_datasets,
        args.model,
        args.K_values,
        config['seeds'],
        args.split,
        config['runs_dir']
    )
    
    if df is None:
        print("⚠ No results found!")
        return
    
    # Save detailed results
    tables_dir = Path(config['tables_dir'])
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    detailed_file = tables_dir / f'heterophilous_{args.model}_split{args.split}_detailed.csv'
    df.to_csv(detailed_file, index=False, float_format='%.6f')
    print(f"✓ Detailed results saved to: {detailed_file}\n")
    
    # Save summary
    summary_file = tables_dir / f'heterophilous_{args.model}_split{args.split}_summary.csv'
    summary.to_csv(summary_file, float_format='%.4f')
    print(f"✓ Summary saved to: {summary_file}\n")
    
    # Print summary
    print("Summary Statistics (mean ± std across seeds):")
    print("=" * 70)
    print(summary)
    print()
    
    # Print per-dataset/K breakdown
    print("\nPer-Configuration Results:")
    print("=" * 70)
    for dataset in heterophilous_datasets:
        dataset_df = df[df['dataset'] == dataset]
        if not dataset_df.empty:
            print(f"\n{dataset}:")
            for K in args.K_values:
                K_df = dataset_df[dataset_df['K'] == K]
                if not K_df.empty:
                    test_mean = K_df['test_acc'].mean()
                    test_std = K_df['test_acc'].std()
                    print(f"  K={K}: Test Acc = {test_mean:.4f} ± {test_std:.4f} ({len(K_df)} seeds)")


if __name__ == '__main__':
    main()
