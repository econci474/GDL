"""
Generate publication-quality results table as PNG.
Aggregates test performance across different model configurations.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def load_probe_results(dataset, model, K, seed):
    """Load test accuracy from probe results."""
    probe_csv = Path(cfg.tables_dir) / f'{dataset}_{model}_K{K}_seed{seed}_probe.csv'
    
    if not probe_csv.exists():
        return None
    
    df = pd.read_csv(probe_csv)
    # Find best k by validation accuracy
    best_k = df.loc[df['val_acc'].idxmax(), 'k']
    test_acc = df.loc[df['k'] == best_k, 'test_acc'].values[0]
    
    return {
        'method': 'Probe',
        'loss_type': 'N/A',
        'k': int(best_k),
        'test_acc': test_acc
    }


def load_classifier_head_results(dataset, model, K, seed, loss_type):
    """Load test accuracy from classifier head test logs."""
    test_log_path = Path(cfg.classifier_heads_dir) / loss_type / dataset / model / f'seed_{seed}' / f'K_{K}' / 'test_log.csv'
    
    if not test_log_path.exists():
        return None
    
    df = pd.read_csv(test_log_path)
    
    #test_log.csv has one row per layer evaluated
    # Take the last row (final layer k=K)
    test_acc = df.iloc[-1]['test_acc']
    
    return {
        'method': 'Classifier Heads',
        'loss_type': loss_type,
        'k': K,  # Final layer
        'test_acc': test_acc
    }


def load_baseline_gnn_results(dataset, model, K, seed):
    """Load test accuracy from baseline GNN training (runs directory)."""
    test_log_path = Path(cfg.runs_dir) / dataset / model / f'seed_{seed}' / f'K_{K}' / 'test_log.csv'
    
    if not test_log_path.exists():
        return None
    
    df = pd.read_csv(test_log_path)
    
    # Take the last row (should be only one row in most cases)
    test_acc = df.iloc[-1]['test_acc']
    
    return {
        'method': 'Baseline GNN',
        'loss_type': 'standard',
        'k': K,
        'test_acc': test_acc
    }


def create_results_table(dataset, model, K_values, seeds, loss_types):
    """Create comprehensive results table."""
    results = []
    
    for seed in seeds:
        for K in K_values:
            # Baseline GNN results (from runs directory)
            baseline_res = load_baseline_gnn_results(dataset, model, K, seed)
            if baseline_res:
                baseline_res['dataset'] = dataset
                baseline_res['model'] = model
                baseline_res['K'] = K
                baseline_res['seed'] = seed
                results.append(baseline_res)
            
            # Probe results (removed - not actual trained models)
            # probe_res = load_probe_results(dataset, model, K, seed)
            # if probe_res:
            #     probe_res['dataset'] = dataset
            #     probe_res['model'] = model
            #     probe_res['K'] = K
            #     probe_res['seed'] = seed
            #     results.append(probe_res)
            
            # Classifier head results
            for loss_type in loss_types:
                ch_res = load_classifier_head_results(dataset, model, K, seed, loss_type)
                if ch_res:
                    ch_res['dataset'] = dataset
                    ch_res['model'] = model
                    ch_res['K'] = K
                    ch_res['seed'] = seed
                    results.append(ch_res)
    
    df = pd.DataFrame(results)
    
    # Reorder columns for readability (exclude 'k' column)
    cols = ['dataset', 'model', 'method', 'loss_type', 'K', 'seed', 'test_acc']
    df = df[cols]
    
    return df


def plot_table_as_image(df, output_path):
    """Render DataFrame as publication-quality table image."""
    
    # Create figure with wider width to accommodate long loss type names
    fig, ax = plt.subplots(figsize=(16, len(df) * 0.4 + 1.5))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare display table
    display_df = df.copy()
    
    # Format test accuracy as percentage with 2 decimal places
    display_df['test_acc'] = display_df['test_acc'].apply(lambda x: f'{x*100:.2f}%')
    
    # Rename columns for publication style
    display_df = display_df.rename(columns={
        'dataset': 'Dataset',
        'model': 'Model',
        'method': 'Method',
        'loss_type': 'Loss Type',
        'K': 'Max Depth',
        'seed': 'Seed',
        'test_acc': 'Test Accuracy'
    })
    
    # Create table with auto column widths
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1],
        colWidths=[0.10, 0.10, 0.15, 0.30, 0.12, 0.08, 0.15]  # Wider Loss Type column, removed Best k
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Header styling
    for i in range(len(display_df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    # Alternate row colors
    for i in range(1, len(display_df) + 1):
        for j in range(len(display_df.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F2F2F2')
            else:
                cell.set_facecolor('white')
            
            # Bold the test accuracy column
            if j == len(display_df.columns) - 1:
                cell.set_text_props(weight='bold')
    
    # Add title
    plt.title('GNN Test Performance Summary', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Results table saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate results table')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--K-values', type=int, nargs='+', default=[3, 8],
                       help='List of K values to include')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0],
                       help='List of seeds to include')
    parser.add_argument('--loss-types', type=str, nargs='+',
                       default=['ce_plus_R_R1.0_hard', 'ce_plus_R_R1.0_smooth', 'class-weighted'],
                       help='List of loss types for classifier heads')
    parser.add_argument('--output-name', type=str, default=None,
                       help='Custom output filename (without extension)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Generating Results Table: {args.dataset} {args.model}")
    print(f"K values: {args.K_values}")
    print(f"Seeds: {args.seeds}")
    print(f"Loss types: {args.loss_types}")
    print(f"{'='*70}\n")
    
    # Create results table
    df = create_results_table(
        args.dataset,
        args.model,
        args.K_values,
        args.seeds,
        args.loss_types
    )
    
    # Print to console
    print("\nResults DataFrame:")
    print(df.to_string(index=False))
    print(f"\n{'='*70}")
    
    # Save CSV
    comparison_dir = Path(cfg.results_dir) / 'comparison_tables'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    if args.output_name:
        csv_path = comparison_dir / f'{args.output_name}.csv'
        png_path = comparison_dir / f'{args.output_name}.png'
    else:
        csv_path = comparison_dir / f'{args.dataset}_{args.model}_test_results_aggregated.csv'
        png_path = comparison_dir / f'{args.dataset}_{args.model}_test_results_table.png'
    
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV saved to: {csv_path}")
    
    # Generate PNG table
    plot_table_as_image(df, png_path)
    
    print(f"\n{'='*70}")
    print(f"✓ Results table generation complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
