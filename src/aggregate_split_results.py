"""Aggregate test results across splits for heterophilous datasets."""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def has_splits(dataset_name, model_name, K, seed, runs_dir):
    """Check if this dataset/model/K/seed uses split-based structure."""
    runs_path = Path(runs_dir) / dataset_name / model_name / f'seed_{seed}' / f'K_{K}'
    split_0_dir = runs_path / 'split_0'
    return split_0_dir.exists()


def load_split_test_accuracies(dataset_name, model_name, K, seed, runs_dir, num_splits=10):
    """Load test accuracies from all splits for a given configuration."""
    test_accs = []
    val_accs = []
    
    for split_idx in range(num_splits):
        log_file = Path(runs_dir) / dataset_name / model_name / f'seed_{seed}' / f'K_{K}' / f'split_{split_idx}' / 'train_log.csv'
        
        if not log_file.exists():
            print(f"    ⚠ Warning: Missing log for split {split_idx}")
            continue
        
        df = pd.read_csv(log_file)
        
        # Get test accuracy at best validation epoch
        best_epoch = df['val_loss'].idxmin()
        test_acc = df.loc[best_epoch, 'test_acc']
        val_acc = df.loc[best_epoch, 'val_acc']
        
        test_accs.append(test_acc)
        val_accs.append(val_acc)
    
    return np.array(test_accs), np.array(val_accs)


def aggregate_results(dataset_name, model_name, runs_dir, seeds, K_values, num_splits=10):
    """Aggregate test results across splits for all K values and seeds."""
    results = []
    
    for K in K_values:
        for seed in seeds:
            # Check if this uses splits
            if not has_splits(dataset_name, model_name, K, seed, runs_dir):
                # Non-split dataset - load single result
                log_file = Path(runs_dir) / dataset_name / model_name / f'seed_{seed}' / f'K_{K}' / 'train_log.csv'
                if log_file.exists():
                    df = pd.read_csv(log_file)
                    best_epoch = df['val_loss'].idxmin()
                    test_acc = df.loc[best_epoch, 'test_acc']
                    val_acc = df.loc[best_epoch, 'val_acc']
                    
                    results.append({
                        'dataset': dataset_name,
                        'model': model_name,
                        'K': K,
                        'seed': seed,
                        'test_acc_mean': test_acc,
                        'test_acc_std': 0.0,
                        'val_acc_mean': val_acc,
                        'val_acc_std': 0.0,
                        'num_splits': 1
                    })
                continue
            
            # Split-based dataset - aggregate across splits
            test_accs, val_accs = load_split_test_accuracies(
                dataset_name, model_name, K, seed, runs_dir, num_splits
            )
            
            if len(test_accs) == 0:
                print(f"  ⚠ No valid splits found for K={K}, seed={seed}")
                continue
            
            results.append({
                'dataset': dataset_name,
                'model': model_name,
                'K': K,
                'seed': seed,
                'test_acc_mean': test_accs.mean(),
                'test_acc_std': test_accs.std(),
                'val_acc_mean': val_accs.mean(),
                'val_acc_std': val_accs.std(),
                'num_splits': len(test_accs)
            })
            
            print(f"  K={K}, seed={seed}: Test Acc = {test_accs.mean():.4f} ± {test_accs.std():.4f} ({len(test_accs)} splits)")
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Aggregate test results across splits')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name')
    parser.add_argument('--num-splits', type=int, default=10,
                       help='Number of splits (default: 10)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Load config
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    print(f"\n{'='*70}")
    print(f"Aggregating Test Results: {args.dataset} - {args.model}")
    print(f"{'='*70}\n")
    
    # Aggregate results
    K_values = list(range(config.get('K_max', 8) + 1))
    df = aggregate_results(
        args.dataset,
        args.model,
        config['runs_dir'],
        config['seeds'],
        K_values,
        args.num_splits
    )
    
    if df.empty:
        print("⚠ No results found!")
        return
    
    # Determine output path
    if args.output:
        output_file = Path(args.output)
    else:
        tables_dir = Path(config['tables_dir'])
        tables_dir.mkdir(parents=True, exist_ok=True)
        output_file = tables_dir / f'{args.dataset}_{args.model}_aggregated_results.csv'
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, float_format='%.6f')
    
    print(f"\n{'='*70}")
    print(f"✓ Results saved to: {output_file}")
    print(f"{'='*70}\n")
    
    # Print summary statistics
    print("Summary by K:")
    print("-" * 70)
    summary = df.groupby('K').agg({
        'test_acc_mean': ['mean', 'std'],
        'val_acc_mean': ['mean', 'std'],
        'num_splits': 'first'
    }).round(4)
    print(summary)
    print()


if __name__ == '__main__':
    main()
