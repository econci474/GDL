"""
Run full pipeline (train, extract, probe, analyze) for multiple K values.

This script automates the complete workflow for a given dataset and model
across all K values from 0 to K_max.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\nWarning: Command failed with return code {result.returncode}")
        print(f"Continuing with next step...\n")
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run full pipeline for multiple K values')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--model', type=str, required=True, help='Model name (GCN, GAT, GraphSAGE)')
    parser.add_argument('--k_min', type=int, default=0, help='Minimum K value')
    parser.add_argument('--k_max', type=int, default=8, help='Maximum K value')
    parser.add_argument('--seeds', type=str, default='all', help='Seeds to run (comma-separated or "all")')
    parser.add_argument('--skip_train', action='store_true', help='Skip training if models exist')
    parser.add_argument('--skip_extract', action='store_true', help='Skip extraction if embeddings exist')
    parser.add_argument('--skip_probe', action='store_true', help='Skip probing if probe results exist')
    
    args = parser.parse_args()
    
    # Determine seeds
    if args.seeds.lower() == 'all':
        seeds = cfg.seeds
    else:
        seeds = [int(s) for s in args.seeds.split(',')]
    
    print(f"\n{'='*60}")
    print(f"Running full pipeline:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  K range: {args.k_min} to {args.k_max}")
    print(f"  Seeds: {seeds}")
    print(f"{'='*60}\n")
    
    # For each K value
    for K in range(args.k_min, args.k_max + 1):
        print(f"\n{'#'*60}")
        print(f"# Processing K={K}")
        print(f"{'#'*60}\n")
        
        for seed in seeds:
            # Check if we should skip based on existing files
            model_path = Path(f"results/runs/{args.dataset}/{args.model}/seed_{seed}/K_{K}/best.pt")
            embeddings_path = Path(f"results/runs/{args.dataset}/{args.model}/seed_{seed}/K_{K}/embeddings.pt")
            probe_path = Path(f"results/tables/{args.dataset}_{args.model}_K{K}_seed{seed}_probe.csv")
            
            # Step 1: Train model
            if not (args.skip_train and model_path.exists()):
                cmd = [
                    'conda', 'run', '-n', 'gdl',
                    'python', '-m', 'src.train_gnn',
                    '--dataset', args.dataset,
                    '--model', args.model,
                    '--K', str(K),
                    '--seed', str(seed)
                ]
                run_command(cmd, f"Training {args.model} on {args.dataset}, K={K}, seed={seed}")
            else:
                print(f"Skipping training for K={K}, seed={seed} (model exists)")
            
            # Step 2: Extract embeddings
            if not (args.skip_extract and embeddings_path.exists()):
                cmd = [
                    'conda', 'run', '-n', 'gdl',
                    'python', '-m', 'src.extract_embeddings',
                    '--dataset', args.dataset,
                    '--model', args.model,
                    '--K', str(K),
                    '--seed', str(seed)
                ]
                run_command(cmd, f"Extracting embeddings for K={K}, seed={seed}")
            else:
                print(f"Skipping extraction for K={K}, seed={seed} (embeddings exist)")
            
            # Step 3: Probe
            if not (args.skip_probe and probe_path.exists()):
                cmd = [
                    'conda', 'run', '-n', 'gdl',
                    'python', '-m', 'src.probe',
                    '--dataset', args.dataset,
                    '--model', args.model,
                    '--K', str(K),
                    '--seed', str(seed)
                ]
                run_command(cmd, f"Probing for K={K}, seed={seed}")
            else:
                print(f"Skipping probing for K={K}, seed={seed} (probe results exist)")
        
        # Step 4: Separability analysis (run once per K for all seeds)
        cmd = [
            'conda', 'run', '-n', 'gdl',
            'python', '-m', 'src.separability_metrics',
            '--dataset', args.dataset,
            '--model', args.model,
            '--K', str(K),
            '--seed', 'all'
        ]
        run_command(cmd, f"Running separability analysis for K={K}, all seeds")
    
    print(f"\n{'='*60}")
    print(f"Pipeline complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
