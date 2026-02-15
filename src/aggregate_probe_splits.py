"""
Aggregate probe results across splits for heterophilous datasets.

For datasets with multiple splits (Minesweeper, Roman-empire), this script:
1. Loads all probe results for a given dataset/model/seed/K
2. Averages metrics across the 10 splits
3. Saves aggregated results in the standard format (without split suffix)

This allows the existing plotting pipeline to work with heterophilous datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

HETEROPHILOUS_DATASETS = ['Minesweeper', 'Roman-empire']


def aggregate_probe_results(dataset_name, model_name, K, seed, config):
    """
    Aggregate probe results across all splits for a given configuration.
    
    Args:
        dataset_name: Dataset name
        model_name: Model name
        K: Number of layers
        seed: Random seed
        config: Configuration dict
    """
    print(f"\nAggregating: {dataset_name} K={K} seed={seed}")
    
    tables_dir = Path(config['tables_dir'])
    arrays_dir = Path(config['results_dir']) / 'arrays'
    
    # Collect all split results
    split_dfs = []
    split_arrays = []
    
    for split_id in range(10):
        # Load CSV table
        csv_path = tables_dir / f'{dataset_name}_{model_name}_K{K}_seed{seed}_split{split_id}_probe.csv'
        if not csv_path.exists():
            print(f"  Warning: Missing {csv_path.name}")
            continue
        
        df = pd.read_csv(csv_path)
        split_dfs.append(df)
        
        # Load NPZ arrays
        npz_path = arrays_dir / f'{dataset_name}_{model_name}_K{K}_seed{seed}_split{split_id}_pernode.npz'
        if npz_path.exists():
            split_arrays.append(dict(np.load(npz_path)))
    
    if not split_dfs:
        print(f"  Error: No split results found!")
        return
    
    print(f"  Found {len(split_dfs)} splits")
    
    # Aggregate CSV metrics (average across splits)
    # Each df has columns: k, val_nll, val_acc, etc.
    aggregated_metrics = {}
    
    for col in split_dfs[0].columns:
        if col == 'k':
            # k is the same across all splits
            aggregated_metrics[col] = split_dfs[0][col].values
        else:
            # Average the metric across splits
            values = np.stack([df[col].values for df in split_dfs], axis=0)  # [num_splits, num_depths]
            aggregated_metrics[col] = values.mean(axis=0)
    
    # Create aggregated DataFrame
    agg_df = pd.DataFrame(aggregated_metrics)
    
    # Save aggregated CSV (without split suffix)
    output_csv = tables_dir / f'{dataset_name}_{model_name}_K{K}_seed{seed}_probe.csv'
    agg_df.to_csv(output_csv, index=False)
    print(f"  ✓ Saved: {output_csv.name}")
    
    #DELETE THIS PART. KEEP NPZ PER SPLIT. DO NOT CONCATENATE
    # Aggregate NPZ arrays (concatenate across splits)
    if split_arrays:
        # For per-node data, we concatenate across splits (each split has different nodes)
        # Keys like 'probs_0', 'probs_1', ..., 'labels', 'val_mask_0', etc.
        aggregated_arrays = {}
        
        # Get all unique keys
        all_keys = set()
        for arr_dict in split_arrays:
            all_keys.update(arr_dict.keys())
        
        for key in all_keys:
            if key == 'k_list':
                # k_list is the same across splits
                aggregated_arrays[key] = split_arrays[0][key]
            else:
                # Concatenate arrays across splits
                arrays = [arr_dict[key] for arr_dict in split_arrays if key in arr_dict]
                if arrays:
                    aggregated_arrays[key] = np.concatenate(arrays, axis=0)
        
        # Save aggregated NPZ (without split suffix)
        output_npz = arrays_dir / f'{dataset_name}_{model_name}_K{K}_seed{seed}_pernode.npz'
        np.savez(output_npz, **aggregated_arrays)
        print(f"  ✓ Saved: {output_npz.name}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate probe results across splits')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (Minesweeper or Roman-empire)')
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--seed', type=str, default='all',
                       help='Seed or "all" for all seeds')
    
    args = parser.parse_args()
    
    if args.dataset not in HETEROPHILOUS_DATASETS:
        print(f"Error: {args.dataset} is not a heterophilous dataset")
        print(f"Expected one of: {HETEROPHILOUS_DATASETS}")
        return
    
    # Convert config module to dict
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    # Determine seeds to process
    if args.seed.lower() == 'all':
        seeds_to_run = [0, 1]  # Only seeds 0-1 were processed
    else:
        seeds_to_run = [int(args.seed)]
    
    # Process all K values (0-8)
    K_values = list(range(9))
    
    print("="*60)
    print(f"Aggregating probe results: {args.dataset}")
    print(f"K values: {K_values}")
    print(f"Seeds: {seeds_to_run}")
    print("="*60)
    
    for seed in seeds_to_run:
        for K in K_values:
            aggregate_probe_results(args.dataset, args.model, K, seed, config)
    
    print("\n" + "="*60)
    print("Aggregation complete!")
    print("="*60)


if __name__ == '__main__':
    main()
